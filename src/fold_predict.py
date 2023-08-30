import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm, trange
from downstream.predict import mlp, fold_dataset

def eval_ss(model, valid_dataloader, device):
    def accuracy(output, target, topk=1):
        accuracy = 0
        for sample_index in range(output.shape[0]):
            ans = torch.topk(output[sample_index], topk)[1]
            accuracy += 1 if ((ans == target[sample_index]).nonzero(as_tuple=True)[0].nelement() != 0) else 0
        return accuracy/(output.shape[0])
    
    num_batches = len(valid_dataloader)
    model.eval()
    cross_entropy = 0
    top_1_accuracy = 0
    top_5_accuracy = 0
    top_10_accuracy = 0

    with torch.no_grad():
        for x, y in valid_dataloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            cross_entropy += F.cross_entropy(y_pred, y)
            top_1_accuracy += accuracy(y_pred, y, 1)
            top_5_accuracy += accuracy(y_pred, y, 5)
            if (y_pred.shape[1]>=10):
                top_10_accuracy += accuracy(y_pred, y, 10)

    print(f"cross_entropy: {cross_entropy/num_batches:>7f}")
    print(f"Test_top_1_accuracy: {top_1_accuracy/num_batches:>7f}")
    print(f"Test_top_5_accuracy: {top_5_accuracy/num_batches:>7f}")
    print(f"Test_top_10_accuracy: {top_10_accuracy/num_batches:>7f}")
    
    return (cross_entropy/num_batches), (top_1_accuracy/num_batches), (top_5_accuracy/num_batches), (top_10_accuracy/num_batches)

def train_ss(model, train_dataloader, device, optimizer):
    size = len(train_dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss_total = F.cross_entropy(y_pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # if batch % 100 == 0:
        #     loss_total, current = loss_total.item(), batch * len(x)
            #print(f"Train_cross_entropy_loss_avg: {loss_total:>7f}  [{current:>5d}/{size:>5d}]")

def main():
    parser = argparse.ArgumentParser('Script for training structure similarity prediction model')

    #input
    parser.add_argument('-mn', '--model_name', type=str, default=None)

    parser.add_argument('-trp', '--train_path', type=str, default='../data/downstream_data/train/')
    parser.add_argument('-vap', '--valid_path', type=str, default='../data/downstream_data/valid/')
    parser.add_argument('-tep', '--test_path', type=str, default='../data/downstream_data/test/')
    parser.add_argument('-rp', '--result_path', type=str, default='../result/fold/')
    parser.add_argument('-d', '--device-id', type=int, default=[0], nargs='+', help='gpu device list, if only cpu then set it None or empty')

    # training parameters
    parser.add_argument('--ss_batch_size', type=int, default=100, help='minibatch size for ss loss (default: 100)')
    parser.add_argument('--epochs', type=int, default=10000, help='number ot epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate (default: 1e-5)')

    #settings
    parser.add_argument('-sc', '--scope_class', type=str, default='fold', help = "One of ['class', 'family', 'superfamily', 'fold']")

    args = parser.parse_args()

    ss_train_dataset = fold_dataset(args.train_path, args.scope_class, args.model_name)
    ss_valid_dataset = fold_dataset(args.valid_path, args.scope_class, args.model_name)
    ss_test_dataset = fold_dataset(args.test_path, args.scope_class, args.model_name)

    embed_dim = ss_train_dataset.get_dim()
    model = mlp(embed_dim = embed_dim, class_dim=ss_train_dataset.get_class_num())

    print(f"Scop class = {args.scope_class}")
    print(f"Scop class class_dim = {ss_train_dataset.get_class_num()}")

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
    batch_size = args.ss_batch_size
    fold_train_dataloader = torch.utils.data.DataLoader(ss_train_dataset
                                                    , batch_size=batch_size
                                                    , shuffle=True
                                                    )
    fold_valid_dataloader = torch.utils.data.DataLoader(ss_valid_dataset
                                                    , batch_size=batch_size
                                                    , shuffle=False
                                                    )
    fold_test_dataloader = torch.utils.data.DataLoader(ss_test_dataset
                                                    , batch_size=batch_size
                                                    , shuffle=False
                                                    )

    optimizer = torch.optim.AdamW(model_methods.parameters(), lr = args.lr)

    eval_cross_entropy_list = []
    eval_top_1_list = []
    eval_top_5_list = []
    eval_top_10_list = []

    test_cross_entropy_list = []
    test_top_1_list = []
    test_top_5_list = []
    test_top_10_list = []

    best_valid_top_1 = 0
    best_cross_entropy = 0
    best_top_1 = 0
    best_top_5 = 0
    best_top_10 = 0
    ## train the model
    print('# training model', file=sys.stderr)
    result_file_path = f"{args.result_path}{args.scope_class}_{args.model_name}"
    output_dir = os.path.dirname(result_file_path)
    os.makedirs(output_dir, exist_ok=True)

    for t in trange(args.epochs):
        #print(f"Epoch {t+1}\n-------------------------------")
        train_ss(model, fold_train_dataloader, device, optimizer)
        if (t+1) % 100 == 0:
            print(f"Epoch {t+1}\n-------------------------------")
            valid_cross_entropy, valid_top_1, valid_top_5, valid_top_10 = eval_ss(model, fold_valid_dataloader, device)
            eval_cross_entropy_list.append(valid_cross_entropy)
            eval_top_1_list.append(valid_top_1)
            eval_top_5_list.append(valid_top_5)
            eval_top_10_list.append(valid_top_10)

            test_cross_entropy, test_top_1, test_top_5, test_top_10 = eval_ss(model, fold_test_dataloader, device)
            test_cross_entropy_list.append(test_cross_entropy)
            test_top_1_list.append(test_top_1)
            test_top_5_list.append(test_top_5)
            test_top_10_list.append(test_top_10)

            if (best_valid_top_1<valid_top_1):
                best_valid_top_1 = valid_top_1
                best_cross_entropy = test_cross_entropy
                best_top_1 = test_top_1
                best_top_5 = test_top_5
                best_top_10 = test_top_10

    with open(result_file_path, 'w') as handle:
        def file_write(name, list):
            handle.write(f'#{name}\n')
            for result in list:
                handle.write(f"{result}\t")
            handle.write('\n')
        file_write('eval_cross_entropy', eval_cross_entropy_list)
        file_write('eval_top_1', eval_top_1_list)
        file_write('eval_top_5', eval_top_5_list)
        file_write('eval_top_10', eval_top_10_list)
        file_write('test_cross_entropy', test_cross_entropy_list)
        file_write('test_top_1', test_top_1_list)
        file_write('test_top_5', test_top_5_list)
        file_write('test_top_10', test_top_10_list)
    
        handle.write(f'best_valid_top_1 = {best_valid_top_1}\n')
        handle.write(f'best_cross_entropy = {best_cross_entropy}\n')
        handle.write(f'best_top_1 = {best_top_1}\n')
        handle.write(f'best_top_5 = {best_top_5}\n')
        handle.write(f'best_top_10 = {best_top_10}\n')

if __name__ == '__main__':
    main()
