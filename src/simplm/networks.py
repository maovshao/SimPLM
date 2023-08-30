import torch
import torch.nn as nn
import esm
from logzero import logger
import torch.nn.functional as F

__all__ = ['SIMPLMNet']


class SIMPLMNet(nn.Module):
    """

    """
    def __init__(self, *, esm_model='../model/esm/esm1b_t33_650M_UR50S.pt', freeze=False, num_fine_tune_layers=None, **kwargs):
        super().__init__()
        logger.info(f'Using {esm_model}, freeze={freeze}, num_fine_tune_layers={num_fine_tune_layers}')
        self.esm, _ = esm.pretrained.load_model_and_alphabet_local(esm_model)
        self.hidden_size = self.esm.args.embed_dim if hasattr(self.esm, 'args') else self.esm.embed_dim
        self.attention_heads = self.esm.args.attention_heads if hasattr(self.esm, 'args') else self.esm.attention_heads
        self.num_layers = self.esm.args.layers if hasattr(self.esm, 'args') else self.esm.num_layers
        self.padding_idx = self.esm.padding_idx
        self.mask_idx = self.esm.mask_idx
        self.freeze = freeze
        if freeze:
            self.esm.requires_grad_(False)
        if num_fine_tune_layers:
            self.esm.embed_tokens.requires_grad_(False)
            if hasattr(self.esm, 'embed_positions'):
                self.esm.embed_positions.requires_grad_(False)
            if hasattr(self.esm, 'emb_layer_norm_before') and self.esm.emb_layer_norm_before:
                self.esm.emb_layer_norm_before.requires_grad_(False)
            for layer in self.esm.layers[:-num_fine_tune_layers]:
                layer.requires_grad_(False)
        self.margin = 0.5

    def forward(self, inputs, task='MLM', **kwargs):
        esm_out = self.esm(inputs.view(-1, inputs.shape[-1]), repr_layers=list(range(self.num_layers + 1)))
        if task == 'MLM':
            return esm_out['logits'].transpose(1, 2)
        esm_out = esm_out['representations'][self.num_layers]
        if self.freeze:
            esm_out = esm_out.detach()
        masks = (inputs != self.padding_idx).view(-1, inputs.shape[-1])
        if task == 'SIM':
            esm_out = esm_out.masked_fill(~masks[..., None], 0.0).sum(dim=1) / masks.sum(dim=1, keepdim=True)
            esm_out = esm_out.view(-1, 3, esm_out.shape[-1])
            dist1 = torch.cdist(esm_out[:, :1], esm_out[:, 1:2]).view(-1)
            dist2 = torch.cdist(esm_out[:, 1:2], esm_out[:, 2:]).view(-1)
            return torch.mean(F.relu(dist1 + self.margin - dist2))
        return esm_out, masks

    def get_params_for_opt(self, lr=1e-5, bert_lr=2e-5, **kwargs):
        return [{'params': self.esm.parameters(), 'lr': bert_lr or lr}]
