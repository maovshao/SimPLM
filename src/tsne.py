import os
import argparse
import torch
import torch.utils.data
from downstream.util import get_scop_result

from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main():
    parser = argparse.ArgumentParser('Script for training structure similarity prediction model')

    #input
    parser.add_argument('-m', '--model_name', type=str, default='onehot')
    parser.add_argument('-sl', '--scop_label', type=str, default='../data/downstream_data/train/fold/scop_lookup.tsv')

    #settings
    parser.add_argument('-o', '--onehot', action='store_true')

    args = parser.parse_args()

    path_list = ["../data/downstream_data/train/fold/embedding/", "../data/downstream_data/valid/fold/embedding/", "../data/downstream_data/test/fold/embedding/"]
    
    mean_esm_result = {}

    for path in path_list:
        embedding_path = f"{path}{args.model_name}"
        mean_esm_result_single = torch.load(embedding_path)
        mean_esm_result.update(mean_esm_result_single)
    
    embed_dim = list(mean_esm_result.values())[0].shape[0]

    label_dataset = get_scop_result(args.scop_label, "class")

    prot_esm_npy = np.empty((len(mean_esm_result),embed_dim))

    for index,i in enumerate(mean_esm_result):
        prot_esm_npy[index] = np.array(mean_esm_result[i].cpu())
    
    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(prot_esm_npy)

    print("Org data dimension is {}, Embedded data dimension is {}".format(prot_esm_npy.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    df_dict = {}
    df_dict['x'] = []
    df_dict['y'] = []
    df_dict['class'] = []

    for index,i in enumerate(mean_esm_result):
        df_dict['x'].append(X_norm[index, 0])
        df_dict['y'].append(X_norm[index, 1])
        df_dict['class'].append(label_dataset[i])
    
    df = pd.DataFrame(dict(X=np.asarray(df_dict['x']), 
        Y=np.asarray(df_dict['y']),
        Class=np.asarray(df_dict['class'])))
    
    sns.scatterplot(data=df, x="X", y="Y", hue="Class", s=10)
    fig_name = f"../result/tsne/t-sne_{args.model_name}.png"
    output_dir = os.path.dirname(fig_name)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(fig_name)
    plt.close()


if __name__ == '__main__':
    main()
