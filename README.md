# Multi-modal Protein Similarity Pre-training via Contrastive and Prompt Learning (SimPLM)


## Requirements

* python==3.10.4
* numpy==1.21.5
* scipy==1.7.3
* scikit-learn==1.1.1
* pytorch==1.12.1
* click==8.0.4
* ruamel.yaml==0.17.21
* biopython==1.78
* tqdm==4.64.0
* logzero==1.7.0

## Train
```bash
torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:29501 --nnodes=1 --nproc_per_node 4 train.py -m train_configures/model.yaml -d train_configures/data.yaml --dist 
```

## Embedding generate
```bash
#onehot
python embedding_generate.py \
-f '../data/downstream_data/test/fold/protein.fasta' \
-mt 'onehot'

#ESM2
python embedding_generate.py \
-f '../data/downstream_data/test/fold/protein.fasta' \
-mt 'esm' \
-mp '../model/esm/esm2_t33_650M_UR50D.pt'

#ESM1b
python embedding_generate.py \
-f '../data/downstream_data/test/fold/protein.fasta' \
-mt 'esm' \
-mp '../model/esm/esm1b_t33_650M_UR50S.pt'

#prot_bert_bfd
python embedding_generate.py \
-f '../data/downstream_data/test/fold/protein.fasta' \
-mt 'prottrans' \
-mp '../model/prottrans/prot_bert_bfd/'

#prot_t5_xl_half_uniref50-enc
python embedding_generate.py \
-f '../data/downstream_data/test/fold/protein.fasta' \
-mt 'prottrans' \
-mp '../model/prottrans/prot_t5_xl_half_uniref50-enc/'
```

## Go predict
```bash
#onehot
python go_predict.py \
-mn 'onehot' \
-m 'mf'

#ESM2
python go_predict.py \
-mn 'esm2_t33_650M_UR50D' \
-m 'mf'

#ESM1b
python go_predict.py \
-mn 'esm1b_t33_650M_UR50S' \
-m 'mf'

#prot_bert_bfd
python go_predict.py \
-mn 'prot_bert_bfd' \
-m 'mf'

#prot_t5_xl_half_uniref50-enc
python go_predict.py \
-mn 'prot_t5_xl_half_uniref50-enc' \
-m 'mf'
```

## Fold predict
```bash
#onehot
python fold_predict.py \
-mn 'onehot' \
-sc 'family'

#ESM2
python fold_predict.py \
-mn 'esm2_t33_650M_UR50D' \
-sc 'family'

#ESM1b
python fold_predict.py \
-mn 'esm1b_t33_650M_UR50S' \
-sc 'family'

#prot_bert_bfd
python fold_predict.py \
-mn 'prot_bert_bfd' \
-sc 'family'

#prot_t5_xl_half_uniref50-enc
python fold_predict.py \
-mn 'prot_t5_xl_half_uniref50-enc' \
-sc 'family'
```

## t-SNE
```bash
#onehot
python tsne.py \
-m 'onehot'
#ESM2
python tsne.py \
-m 'esm2_t33_650M_UR50D'
#ESM1b
python tsne.py \
-m 'esm1b_t33_650M_UR50S'
#prot_bert_bfd
python tsne.py \
-m 'prot_bert_bfd'
#prot_t5_xl_half_uniref50-enc
python tsne.py \
-m 'prot_t5_xl_half_uniref50-enc'
```