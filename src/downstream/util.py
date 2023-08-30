import joblib
from pathlib import Path
import os
import torch
import numpy as np
import json
from tqdm import tqdm, trange
from logzero import logger
import scipy.sparse as ssp
from scipy.sparse import find
from Bio import SeqIO
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer

def get_pid_list(pid_list_file):
    try:
        with open(pid_list_file) as fp:
            return [line.split()[0] for line in fp]
    except TypeError:
        return pid_list_file

def get_index_protein_dic(protein_list):
    protein_dic = {}
    for index,protein in enumerate(protein_list):
        protein_dic[index] = protein
    return protein_dic

def get_protein_index_dic(protein_list):
    protein_dic = {}
    for index,protein in enumerate(protein_list):
        protein_dic[protein] = index
    return protein_dic

def get_go_list(pid_go_file, pid_list):
    if pid_go_file is not None:
        pid_go = defaultdict(list)
        with open(pid_go_file) as fp:
            for line in fp:
                pid_go[(line_list:=line.split())[0]].append(line_list[1])
        return [pid_go[pid_] for pid_ in pid_list]
    else:
        return None

def get_mlb(mlb_path: Path, labels=None, **kwargs) -> MultiLabelBinarizer:
    if mlb_path.exists():
        return joblib.load(mlb_path)
    mlb = MultiLabelBinarizer(sparse_output=True, **kwargs)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb

def get_pid_go_mat(pid_go, pid_list, go_list):
    go_mapping = {go_: i for i, go_ in enumerate(go_list)}
    r_, c_, d_ = [], [], []
    for i, pid_ in enumerate(pid_list):
        if pid_ in pid_go:
            for go_ in pid_go[pid_]:
                if go_ in go_mapping:
                    r_.append(i)
                    c_.append(go_mapping[go_])
                    d_.append(1)
    return ssp.csr_matrix((d_, (r_, c_)), shape=(len(pid_list), len(go_list)))

def get_pid_go_sc_mat(pid_go_sc, pid_list, go_list):
    sc_mat = np.zeros((len(pid_list), len(go_list)))
    for i, pid_ in enumerate(pid_list):
        if pid_ in pid_go_sc:
            for j, go_ in enumerate(go_list):
                sc_mat[i, j] = pid_go_sc[pid_].get(go_, -1e100)
    return sc_mat

def read_fasta(fn_fasta):
    prot2seq = {}
    with open(fn_fasta) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq)
            prot = record.id
            prot2seq[prot] = seq
    return list(prot2seq.keys()), prot2seq

def get_scop_result(prefilter_result, scop_class):
    protein_fold_dic = {}
    with open(prefilter_result) as fp:
        for line in fp:
            line_list = line.strip().split(' ')
            protein = line_list[0]
            label_list = line_list[1].split('.')
            if (scop_class == 'class'):
                label = label_list[0]
            elif (scop_class == 'fold'):
                label = label_list[0]+'.'+label_list[1]
            elif (scop_class == 'superfamily'):
                label = label_list[0]+'.'+label_list[1]+'.'+label_list[2]
            elif (scop_class == 'family'):
                label = label_list[0]+'.'+label_list[1]+'.'+label_list[2]+'.'+label_list[3]
            protein_fold_dic[protein]=label
    return protein_fold_dic

def output_res(res_path, pid_list, go_list, sc_mat):
    res_path = Path(res_path)
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, 'w') as fp:
        for pid_, sc_ in zip(pid_list, sc_mat):
            for go_, s_ in zip(go_list, sc_):
                print(pid_, go_, s_, sep='\t', file=fp)

def get_pid_go(pid_go_file):
    if pid_go_file is not None:
        pid_go = defaultdict(list)
        with open(pid_go_file) as fp:
            for line in fp:
                pid_go[(line_list:=line.split('\t'))[0]].append(line_list[1])
        return dict(pid_go)
    else:
        return None

def get_pid_go_sc(pid_go_sc_file):
    pid_go_sc = defaultdict(dict)
    with open(pid_go_sc_file) as fp:
        for line in fp:
            pid_go_sc[line_list[0]][line_list[1]] = float((line_list:=line.split('\t'))[2])
    return dict(pid_go_sc)