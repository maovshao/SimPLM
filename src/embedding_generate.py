#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os
import argparse
import torch
import torch.utils.data
from downstream.embedding_generate import embedding_generate

def main():
    parser = argparse.ArgumentParser('Script for training structure similarity prediction model')

    #input
    parser.add_argument('-f', '--fasta', type=str, default=None)
    parser.add_argument('-mt', '--model_type', type=str, default=None, help="one of ['onehot', 'esm', 'simplm', prottrans]")
    parser.add_argument('-mp', '--model_path', type=str, default=None)
    parser.add_argument('--nogpu', action='store_true')

    args = parser.parse_args()
    
    mean_esm_result = embedding_generate(args.fasta, args.model_type, args.model_path, args.nogpu)
    
    output_path = ''.join([x+'/' for x in args.fasta.split('/')[:-1]]) + 'embedding/'
    if args.model_type == "onehot":
        model_name = 'onehot'
    elif args.model_type == "esm":
        model_name = args.model_path.split('/')[-1].split('.pt')[0]
    else:
        model_name = args.model_path.split('/')[-2]

    embedding_path = f"{output_path}{model_name}"
    os.makedirs(output_path, exist_ok=True)
    torch.save(mean_esm_result, embedding_path)

if __name__ == '__main__':
    main()
