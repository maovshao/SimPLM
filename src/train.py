import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
import click
import numpy as np
import scipy.sparse as ssp
import torch
import torch.distributed as dist
from pathlib import Path
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from logzero import logger

from simplm.data_utils import get_seq, get_sim
from simplm.datasets import SIMPLMDataset
from simplm.samplers import OrderedDistributedSampler
from simplm.models import SIMPLMModel


__all__ = []


def get_dataloader(inputs, targets, batch_size, enable_dist=False, shuffle=False, is_training=False, **kwargs):
    return DataLoader(d_:=SIMPLMDataset(inputs=inputs, targets=targets, is_training=is_training, **kwargs),
                      batch_size=batch_size,
                      shuffle=shuffle if not enable_dist else None,
                      num_workers=4 if not enable_dist else 0,
                      sampler=OrderedDistributedSampler(d_, shuffle=shuffle) if enable_dist else None)


@click.command()
@click.option('-d', '--data-cnf', type=Path, help='Path of dataset configure yaml.')
@click.option('-m', '--model-cnf', type=Path, help='Path of model configure yaml.')
@click.option('--dist', 'enable_dist', is_flag=True)
@click.option('-a', '--amp', 'enable_amp', is_flag=True)
def main(data_cnf, model_cnf, enable_dist, enable_amp):
    if enable_dist:
        dist.init_process_group(backend='nccl')
        logger.info(f'Using DDP with rank {dist.get_rank()}')
        if dist.get_rank() > 0:
            logger.setLevel(100)
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(data_cnf), yaml.load(model_cnf)
    model_name = model_cnf['name']
    model_cnf.setdefault('model', {})
    model_cnf.setdefault('dataset', {})
    model_cnf['model']['model_path'] = Path(data_cnf['model_path']) / f'{model_name}.pt'
    batch_size = model_cnf['batch_size']
    model_cnf['model']['enable_amp'] = enable_amp
    if enable_dist:
        model_cnf['model']['local_rank'] = local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        batch_size //= dist.get_world_size()
    logger.info(f'Batch size of Model: {model_cnf["batch_size"]}, '
                f'Batch size of DataLoader: {batch_size}, '
                f'World size: {1 if not enable_dist else dist.get_world_size()}')
    
    model = SIMPLMModel(**model_cnf['model'])

    mlm_file = data_cnf['train']['mlm']
    sim_file = data_cnf['train']['sim']

    seq_id_list, seq_fasta_list = get_seq(mlm_file)
    similarity_mat = get_sim(sim_file, seq_id_list)

    train_x = {'MLM': seq_fasta_list,
               'SIM': seq_fasta_list}
    train_y = {'MLM': None, 'SIM': similarity_mat}
    train_loader = get_dataloader(train_x, train_y, batch_size, enable_dist=enable_dist,
                                  is_training=True, shuffle=True, **model_cnf['dataset'])

    model.train(train_loader, **model_cnf['train'])

if __name__ == '__main__':
    main()
