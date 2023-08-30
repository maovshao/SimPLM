import numpy as np
import torch
import torch.distributed as dist
import esm
from numpy.random import MT19937, RandomState, SeedSequence
from torch.utils.data import Dataset
from tqdm import tqdm
from logzero import logger

__all__ = ['SIMPLMDataset']


class SIMPLMDataset(Dataset):
    """

    """
    def __init__(self, inputs, targets, is_training=False, esm_model='../model/esm/esm1b_t33_650M_UR50S.pt', max_len=1000,
                 mask_rate=0.15, **kwargs):
        self.max_len, self.is_training = max_len, is_training
        _, self.alphabet = esm.pretrained.load_model_and_alphabet_local(esm_model)
        self.inputs, self.targets, self.mask_rate = {}, targets, mask_rate
        for task in ('MLM', 'SIM'):
            if task in inputs:
                self.convert_to_tensor(inputs, task)

    def __len__(self):
        return self.inputs['MLM'].shape[0]

    def __getitem__(self, item):
        inputs = {'MLM': self.inputs['MLM'][item]}
        if self.is_training:
            targets = {'MLM': self.targets['MLM'][item]}

            task = 'SIM'
            while True:
                x1 = np.random.randint(self.targets[task].shape[0])
                if len(self.targets[task][x1].indices) > 0:
                    break

            x2 = np.random.randint(len(self.targets[task][x1].indices))
            x2 = self.targets[task][x1].indices[x2]

            with_pair = set(self.targets[task][x1].indices)
            without_pair = set(range(self.targets[task].shape[1])) - with_pair
            x3 = np.random.choice(list(without_pair))

            test1 = self.targets[task][x1, x2]
            if (test1 <= 0):
                print("Bad positive sample!\n")
            test2 = self.targets[task][x1, x3]
            if (test2 > 0):
                print("Bad negitive sample!\n")

            inputs[task] = torch.vstack([self.inputs[task][x2], self.inputs[task][x1], self.inputs[task][x3]])
            targets[task] = ""

            return inputs, targets
        else:
            return inputs

    def convert_to_tensor(self, inputs, task):
        logger.info(f'Convert acids of sequences to ids(task:{task}).')
        map_ = {x: i for i, x in enumerate(self.alphabet.all_toks)}
        self.inputs[task] = torch.as_tensor([[self.alphabet.cls_idx] +
                                             [map_.get(x, self.alphabet.unk_idx) for x in d[:self.max_len]] +
                                             [self.alphabet.eos_idx] +
                                             [self.alphabet.padding_idx] * (self.max_len - len(d))
                                             for d in tqdm(inputs[task], desc=f'Converting(task:{task})', leave=False,
                                                           disable=dist.is_initialized() and dist.get_rank() > 0)])
        if task == 'MLM':
            inputs_, self.targets[task] = self.inputs[task], self.inputs[task].clone().numpy()
            rs = RandomState(MT19937(SeedSequence(621668))) if not self.is_training else np.random
            masked_idx = ((inputs_ != self.alphabet.cls_idx) &
                          (inputs_ != self.alphabet.eos_idx) &
                          (inputs_ != self.alphabet.padding_idx) &
                          (torch.as_tensor(rs.rand(*inputs_.shape)) < self.mask_rate))
            if self.is_training:
                r_ = torch.as_tensor(rs.rand(*inputs_.shape))
                inputs_[masked_idx & (r_ < 0.8)] = self.alphabet.mask_idx
                inputs_[masked_idx & (r_ > 0.9)] = torch.as_tensor(rs.randint(s_ := len(self.alphabet.prepend_toks),
                                                                              s_ + len(self.alphabet.standard_toks),
                                                                              (masked_idx & (r_ > 0.9)).sum().item()))
            else:
                inputs_[masked_idx] = self.alphabet.mask_idx
            self.targets[task][~masked_idx] = -100
