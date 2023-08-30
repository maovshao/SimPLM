import Bio.SeqIO
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
import scipy.sparse as ssp

__all__ = ['get_seq','get_sim']


def get_seq(fasta_files):
    seq_id_list = []
    seq_fasta_list = []

    for fasta_file in fasta_files:
        for seq in tqdm(Bio.SeqIO.parse(fasta_file, 'fasta'), leave=False,
                        disable=dist.is_initialized() and dist.get_rank() > 0):
            seq_id_list.append(seq.id)
            seq_fasta_list.append(str(seq.seq))

    return seq_id_list, seq_fasta_list

def get_sim(similarity_files, seq_id_list):
    num_proteins = len(seq_id_list)
    similarity_mat = ssp.lil_matrix((num_proteins, num_proteins), dtype=np.float32)

    for similarity_file in similarity_files:
        with open(similarity_file, 'r') as file:
            for line in file:
                protein1, protein2, similarity = line.strip().split('\t')
                similarity = float(similarity)
                if (similarity > 0.5) and (protein1 in seq_id_list) and (protein2 in seq_id_list):
                        index1 = seq_id_list.index(protein1)
                        index2 = seq_id_list.index(protein2)
                        similarity_mat[index1, index2] = 1
                        similarity_mat[index2, index1] = 1
    return similarity_mat.tocsr()