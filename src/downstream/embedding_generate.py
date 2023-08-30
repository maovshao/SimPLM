from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F

from esm import FastaBatchedDataset, pretrained
from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertModel, BertTokenizer
from simplm.models import simplm

def embedding_generate(fasta, model_type, model_path = None, nogpu=False):
    mean_embedding_dic = {}
    if model_type == "prottrans":
        mean_embedding_dic = prottrans_get_embeddings(fasta, model_path, nogpu)
        return mean_embedding_dic

    _, alphabet = pretrained.load_model_and_alphabet('../model/esm/esm1b_t33_650M_UR50S.pt')
    dataset = FastaBatchedDataset.from_file(fasta)
    batches = dataset.get_batch_indices(16384, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
    )
    print(f"Read {fasta} with {len(dataset)} sequences")

    if model_type != "onehot":
        if model_type == "esm":
            model, alphabet = pretrained.load_model_and_alphabet(model_path)
        elif model_type == "simplm":
            model_path = Path(model_path)
            simplm_model = simplm(model_path = model_path, esm_model='../model/esm/esm1b_t33_650M_UR50S.pt')
            simplm_model.load_model()
            model = simplm_model.model.esm

        model.eval()
        if torch.cuda.is_available() and not nogpu:
            model = model.cuda()
            print("Transferred model to GPU")
        num_layers = model.args.layers if hasattr(model, 'args') else model.num_layers

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(tqdm(data_loader)):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and not nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            if model_type != "onehot":
                out = model(toks, repr_layers=[num_layers], return_contacts=False)["representations"][num_layers]
            else:
                out = F.one_hot(toks, num_classes = len(alphabet.all_toks)).to(torch.float32)

            for i, label in enumerate(labels):
                #get mean embedding
                #mean_embedding_dic[label] = out[i, 1 : len(strs[i]) + 1].mean(0).clone().cpu()
                mean_embedding_dic[label] = out[i, 1 : len(strs[i]) + 1].mean(0).clone()
    return mean_embedding_dic

def prottrans_get_embeddings(seq_path, 
                   model_dir,
                   nogpu,
                   max_residues=2000, # number of cumulative residues per batch
                   max_seq_len=1022, # max length after which we switch to single-sequence processing to avoid OOM
                   max_batch=50 # max number of sequences per single batch
                   ):
    def prottrans_get_t5_model(model_dir, transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"):
        print("Loading: {}".format(transformer_link))
        if model_dir is not None:
            print("##########################")
            print("Loading cached model from: {}".format(model_dir))
            print("##########################")
        model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=model_dir)
        model.full() if device=='cpu' else model.half() # only cast to full-precision if no GPU is available

        model = model.to(device)
        model = model.eval()
        tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)
        return model, tokenizer
    
    def prottrans_get_bert_model(model_dir, transformer_link = "Rostlab/prot_bert_bfd"):
        print("Loading: {}".format(transformer_link))
        if model_dir is not None:
            print("##########################")
            print("Loading cached model from: {}".format(model_dir))
            print("##########################")
        model = BertModel.from_pretrained(transformer_link, cache_dir=model_dir)

        model = model.to(device)
        model = model.eval()
        tokenizer = BertTokenizer.from_pretrained(transformer_link, do_lower_case=False)
        return model, tokenizer

    def prottrans_read_fasta( fasta_path ):
        '''
            Reads in fasta file containing multiple sequences.
            Returns dictionary of holding multiple sequences or only single 
            sequence, depending on input file.
        '''
        
        sequences = dict()
        with open( fasta_path, 'r' ) as fasta_f:
            for line in fasta_f:
                # get uniprot ID from header and create new entry
                if line.startswith('>'):
                    uniprot_id = line.replace('>', '').strip()
                    # replace tokens that are mis-interpreted when loading h5
                    uniprot_id = uniprot_id.replace("/","_").replace(".","_")
                    sequences[ uniprot_id ] = ''
                else:
                    # repl. all whie-space chars and join seqs spanning multiple lines
                    sequences[ uniprot_id ] += ''.join( line.split() ).upper().replace("-","") # drop gaps and cast to upper-case
                    
        return sequences

    device = torch.device('cuda:0' if torch.cuda.is_available() and not nogpu else 'cpu')
    print("Using device: {}".format(device))

    seq_dict = dict()
    emb_dict = dict()

    # Read in fasta
    seq_dict = prottrans_read_fasta(seq_path)
    model_name = model_dir.split('/')[-2]
    if model_name == 'prot_bert_bfd':
        model, vocab = prottrans_get_bert_model(model_dir)
    elif model_name == 'prot_t5_xl_half_uniref50-enc':
        model, vocab = prottrans_get_t5_model(model_dir)
    else:
        print("No such model!")

    print('########################################')
    print('Example sequence: {}\n{}'.format( next(iter(
            seq_dict.keys())), next(iter(seq_dict.values()))) )
    print('########################################')
    print('Total number of sequences: {}'.format(len(seq_dict)))

    avg_length = sum([ len(seq) for _, seq in seq_dict.items()]) / len(seq_dict)
    n_long     = sum([ 1 for _, seq in seq_dict.items() if len(seq)>max_seq_len])
    seq_dict   = sorted( seq_dict.items(), key=lambda kv: len( seq_dict[kv[0]] ), reverse=True )
    
    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))
    
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(tqdm(seq_dict),1):
        seq = seq.replace('U','X').replace('Z','X').replace('O','X')
        seq_len = len(seq)
        # seq = ' '.join(list(seq))
        seq = ' '.join(list(seq[:max_seq_len]))  # 裁剪序列到指定的最大长度
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = vocab.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
            
            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={}). Try lowering batch size. ".format(pdb_id, seq_len) +
                      "If single sequence processing does not work, you need more vRAM to process your protein.")
                continue
            
            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = min(seq_lens[batch_idx], max_seq_len)
                #s_len = seq_lens[batch_idx]
                # slice-off padded/special tokens
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                
                emb = emb.mean(dim=0)
            
                if len(emb_dict) == 0:
                    print("Embedded protein {} with length {} to emb. of shape: {}".format(
                        identifier, s_len, emb.shape))

                emb_dict[ identifier ] = emb.clone().squeeze().float()

    print('\n############# STATS #############')
    print('Total number of embeddings: {}'.format(len(emb_dict)))
    return emb_dict
