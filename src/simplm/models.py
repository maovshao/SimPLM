import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.cuda.amp as amp
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from logzero import logger

from simplm.networks import *

class SIMPLMModel(object):
    def __init__(self, *, model_path: Path, local_rank=None, enable_amp=False, **kwargs):
        assert local_rank is None or dist.is_initialized()
        self.model = self.network = SIMPLMNet(enable_amp=enable_amp, **kwargs).cuda(local_rank)
        self.dp_network = nn.DataParallel(self.network) if local_rank is None else DDP(self.network,
                                                                                       device_ids=[local_rank],
                                                                                       output_device=local_rank,
                                                                                       find_unused_parameters=True)
        self.model_path = model_path
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.loss_fn = self.optimizer = None
        self.training_state = None
        self.local_rank = local_rank
        self.enable_amp, self.scaler = enable_amp, amp.GradScaler(enabled=enable_amp)
        logger.info(f'Using AMP: {enable_amp}')

    def get_optimizer(self, lr=1e-5, lr_params=(), eps=1e-8, weight_decay=0.0, **kwargs):
        logger.info(F'lr={lr}, lr_params={lr_params}, weight_decay={weight_decay}')
        self.optimizer = torch.optim.AdamW(self.model.get_params_for_opt(lr, **dict(lr_params)), eps=eps,
                                           weight_decay=weight_decay, **kwargs)

    def get_loss_fn(self, **kwargs):
        self.loss_fn = {'MLM': nn.CrossEntropyLoss()}

    def get_scores(self, inputs, **kwargs):
        return self.dp_network(inputs, **kwargs).float()

    def loss_and_backward(self, inputs, targets, task='MLM', weight = 1, **kwargs):
        if (task=='MLM'):
            scores = self.get_scores(inputs[task], task=task, **kwargs)
            loss = self.loss_fn[task](scores, targets[task].cuda(self.local_rank))
        else:
            loss = self.get_scores(inputs[task], task=task, **kwargs)
        loss = loss * weight
        self.scaler.scale(loss).backward()
        return loss.item()

    def train_step(self, inputs, targets, task_weight, **kwargs):
        self.training_state['trained_step'] += 1
        self.model.train()
        loss = {task: self.loss_and_backward(inputs, targets, task=task, weight = task_weight[task], **kwargs) 
                if (task_weight[task]>0) else 0.0
                for task in ('MLM', 'SIM')}
        self.update()
        return loss

    def update(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def init_training(self, train_loader, opt_options=(), loss_options=(), num_epochs=10):
        self.training_state = {'num_training_steps': len(train_loader) * num_epochs, 'epoch_idx': 0,
                               'trained_step': 0, 'best': -np.inf}
        self.get_optimizer(**dict(opt_options)), self.get_loss_fn(**dict(loss_options))

    def train(self, train_loader: DataLoader, opt_options=(), loss_options=(),
              num_epochs=10, valid_step=100, task_weight = {'MLM':1, 'SIM':1}, **kwargs):
        self.init_training(train_loader, opt_options, loss_options, num_epochs)
        bs_ = (dist.get_world_size() if self.local_rank is not None else 1) * train_loader.batch_size
        bar_tot = valid_step * bs_
        training_bar = tqdm(desc='Training', leave=False, dynamic_ncols=True, total=bar_tot,
                            disable=self.local_rank is not None and dist.get_rank() > 0)
        for self.training_state['epoch_idx'] in range(num_epochs):
            for inputs, targets in train_loader:
                train_loss = self.train_step(inputs, targets, task_weight)
                training_bar.update(bs_)
                if valid_step is not None and self.training_state['trained_step'] % valid_step == 0:
                    training_bar.close()
                    logger.info(f'Epoch: {self.training_state["epoch_idx"]}, '
                    f'MLM Loss: {train_loss["MLM"]:.5f}, '
                    f'SIM Loss: {train_loss["SIM"]:.5f}')
                    training_bar = tqdm(desc='Training', leave=False, dynamic_ncols=True, total=bar_tot,
                                        disable=self.local_rank is not None and dist.get_rank() > 0)
            self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        logger.info(f'loading model from {self.model_path}')
        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'), strict=False)
