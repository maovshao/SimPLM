import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import scipy.sparse as ssp
from tqdm import tqdm, trange
from downstream.predict import mlp, go_dataset
from downstream.util import get_mlb, read_fasta, get_go_list, output_res
from downstream.evaluation import fmax, aupr, evaluate_metrics

def eval_ss(model, valid_dataloader, device):
    model.eval()
    y = []
    y_pred = []

    with torch.no_grad():
        for x_, y_ in valid_dataloader:
            x_ = x_.to(device)
            y_ = y_.to(device)
            y_pred_ = model(x_)
            y.append(y_)
            y_pred.append(y_pred_)

        y = torch.cat(y)
        y_pred = torch.cat(y_pred)

        y_pred = torch.sigmoid(y_pred).cpu().numpy()
        y = ssp.csr_matrix(y.cpu().numpy())

        (fmax_, t_) = fmax(y, y_pred)
        aupr_ = aupr(y.toarray().flatten(), y_pred.flatten())

        print(f"F-max: {fmax_:>7f}")
        print(f"AUPR: {aupr_:>7f}")
    
    return fmax_, y_pred
    #return fmax_, y.toarray()

def train_ss(model, train_dataloader, device, optimizer):
    size = len(train_dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss_total = F.binary_cross_entropy_with_logits(y_pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_total, current = loss_total.item(), batch * len(x)
            print(f"Train_cross_entropy_loss_avg: {loss_total:>7f}  [{current:>5d}/{size:>5d}]")

def main():
    parser = argparse.ArgumentParser('Script for training structure similarity prediction model')

    #input
    parser.add_argument('-mn', '--model_name', type=str, default=None)

    parser.add_argument('-trp', '--train_path', type=str, default='../data/downstream_data/train/')
    parser.add_argument('-vap', '--valid_path', type=str, default='../data/downstream_data/valid/')
    #parser.add_argument('-tep', '--test_path', type=str, default='../data/downstream_data/test/')
    parser.add_argument('-tep', '--test_path', type=str, default='../data/downstream_data/test/1000/')
    #parser.add_argument('-rep', '--result_path', type=str, default='../result/go/')
    parser.add_argument('-rep', '--result_path', type=str, default='../result/go/1000/')

    parser.add_argument('-d', '--device-id', type=int, default=[0], nargs='+', help='gpu device list, if only cpu then set it None or empty')

    # training parameters
    parser.add_argument('--ss_batch_size', type=int, default=100, help='minibatch size for ss loss (default: 100)')
    parser.add_argument('--epochs', type=int, default=500, help='number ot epochs (default: 500)')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate (default: 1e-5)')

    #settings
    parser.add_argument('-m', '--mode', type=str, help = "One of ['mf', 'bp', 'cc']")
    args = parser.parse_args()

    ## get_mlb
    fasta = f"{args.train_path}{args.mode}/protein.fasta"
    go_path = f"{args.train_path}{args.mode}/go.txt"
    mlb_path = f"{args.train_path}{args.mode}/label.mlb"
    _, protein_list = read_fasta(fasta)
    go = get_go_list(go_path, protein_list)
    mlb = get_mlb(Path(mlb_path), go)
    labels_num = len(mlb.classes_)

    train_dataset = go_dataset(args.train_path, args.mode, mlb, args.model_name)
    valid_dataset = go_dataset(args.valid_path, args.mode, mlb, args.model_name)
    test_dataset = go_dataset(args.test_path, args.mode, mlb, args.model_name)

    embed_dim = train_dataset.get_dim()
    model = mlp(embed_dim = embed_dim, class_dim=labels_num)

    ## set the device
    if (args.device_id == None or args.device_id == []):
        print("None of GPU is selected.")
        device = "cpu"
        model.to(device)
        model_methods = model
    else:
        if torch.cuda.is_available()==False:
            print("GPU selected but none of them is available.")
            device = "cpu"
            model.to(device)
            model_methods = model
        else:
            print("We have", torch.cuda.device_count(), "GPUs in total! We will use as you selected")
            model = nn.DataParallel(model, device_ids = args.device_id)
            device = f'cuda:{args.device_id[0]}'
            model.to(device)
            model_methods = model.module

    print(f'# training with esm_ss_predict_tri: ss_batch_size={args.ss_batch_size}, epochs={args.epochs}, lr={args.lr}')
  
    # iterators for the ss data
    train_dataloader = torch.utils.data.DataLoader(train_dataset
                                                    , batch_size=args.ss_batch_size
                                                    , shuffle=True
                                                    )
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset
                                                    , batch_size=args.ss_batch_size
                                                    , shuffle=False
                                                    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset
                                                    , batch_size=args.ss_batch_size
                                                    , shuffle=False
                                                    )

    optimizer = torch.optim.AdamW(model_methods.parameters(), lr = args.lr)

    best_valid_fmax = 0
    valid_fmax_list = []
    test_fmax_list = []

    ## train the model
    print('# training model', file=sys.stderr)
    predict_result_file_path = f"{args.result_path}{args.mode}_{args.model_name}.txt"
    fasta = f"{args.test_path}{args.mode}/protein.fasta"
    test_protein_list, _ = read_fasta(fasta)

    for t in trange(args.epochs):
        #print(f"Epoch {t+1}\n-------------------------------")
        train_ss(model, train_dataloader, device, optimizer)
        if (t+1) % 10 == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            valid_fmax, _ = eval_ss(model, valid_dataloader, device)

            if (best_valid_fmax<valid_fmax):
                best_valid_fmax = valid_fmax
                print(f"---------------Testing----------------")
                test_famx, predict_result = eval_ss(model, test_dataloader, device)
                valid_fmax_list.append(valid_fmax)
                test_fmax_list.append(test_famx)
                output_res(predict_result_file_path, test_protein_list, mlb.classes_, predict_result)
                pid_go = f"{args.test_path}{args.mode}/go.txt"
                pid_go_sc = predict_result_file_path
                (fmax_, t_), aupr_, _, _ = evaluate_metrics(pid_go, pid_go_sc, if_m_aupr = False)
                print(F'Fmax: {fmax_:.3f} {t_:.2f}', F'AUPR: {aupr_:.3f}')
    
    print(f"GO mode = {args.mode}")
    print(f"GO label num = {labels_num}")
    print(f"valid_fmax_list: {valid_fmax_list}")
    print(f"test_fmax_list: {test_fmax_list}")

    pid_go = f"{args.test_path}{args.mode}/go.txt"
    pid_go_sc = predict_result_file_path
    (fmax_, t_), aupr_, _, _ = evaluate_metrics(pid_go, pid_go_sc, if_m_aupr = False)
    print(F'Fmax: {fmax_:.3f} {t_:.2f}', F'AUPR: {aupr_:.3f}')

if __name__ == '__main__':
    main()
