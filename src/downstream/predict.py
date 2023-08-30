import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from downstream.util import read_fasta, get_go_list, get_scop_result

class mlp(nn.Module):
    def __init__(self, embed_dim, class_dim):
        super(mlp, self).__init__()
        self.linear1 = nn.Linear(embed_dim, 480)
        self.linear2 = nn.Linear(480, 480)
        self.linear3 = nn.Linear(480, 480)
        self.classifer = nn.Linear(480, class_dim)

        self.dropout = nn.Dropout(0.5)

    def load_pretrained(self, simplm_path):
        #load to cpu at first, and then tranfer according to device_id
        state_dict = torch.load(simplm_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict)

    def forward(self, z):
        outputs = self.dropout(F.relu(self.linear1(z)))
        outputs = self.dropout(F.relu(self.linear2(outputs)))
        outputs = self.dropout(F.relu(self.linear3(outputs)))
        outputs = self.classifer(outputs)
        return outputs

class go_dataset:
    def __init__(self, path, mode, mlb, model_name):
        go_path = f"{path}{mode}/go.txt"
        fasta_path = f"{path}{mode}/protein.fasta"
        embedding_path = f"{path}{mode}/embedding/{model_name}"

        if (os.path.exists(embedding_path)):
            mean_esm_result = torch.load(embedding_path)
        else:
            print(f"Embedding in {embedding_path} not found!")

        protein_list, _ = read_fasta(fasta_path)
        #go = get_go_list(go_path, list(mean_esm_result.keys())) # must use the order in fasta/pid, or it will be different with the output list order(function: output_res)
        go = get_go_list(go_path, protein_list)
        labels_num = len(mlb.classes_)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            go_label = mlb.transform(go).astype(np.float32)

        print(f'# loading esm, proetin_num={len(mean_esm_result)}')
        print(f"GO label num = {labels_num}")
        
        self.x = []
        self.y = []

        #for index, protein in enumerate(mean_esm_result): # must use the order in fasta/pid, or it will be different with the output list order(function: output_res)
        for index, protein in enumerate(protein_list):
            self.x.append(mean_esm_result[protein])
            self.y.append(torch.tensor(go_label[index].toarray()).squeeze())
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def get_dim(self):
        return(int(self.x[0].shape[0]))

class fold_dataset:
    def __init__(self, path, scop_class, model_name):
        scop_csv_filename = f"../data/downstream_data/train/fold/scop_lookup.tsv"
        fasta_path = f"{path}fold/protein.fasta"
        embedding_path = f"{path}fold/embedding/{model_name}"

        if (os.path.exists(embedding_path)):
            mean_esm_result = torch.load(embedding_path)
        else:
            print(f"Embedding in {embedding_path} not found!")

        protein_list, _ = read_fasta(fasta_path)
        protein_fold_dic = get_scop_result(scop_csv_filename, scop_class)

        fold_set = set()
        for protein in protein_fold_dic:
            fold_set.add(protein_fold_dic[protein])
        
        fold_dic = {}
        for index, fold in enumerate(fold_set):
            fold_dic[fold] = index
        
        num_classes = len(fold_set)
        
        self.x = []
        self.y = []

        for protein in protein_list:
            self.x.append(mean_esm_result[protein])
            self.y.append(torch.tensor(fold_dic[protein_fold_dic[protein]]))

        print(f'# class_num = {num_classes}')
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def get_dim(self):
        return(int(self.x[0].shape[0]))
    
    def get_class_num(self):
        return self.num_classes