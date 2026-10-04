import copy

import torch
from torch import Tensor, nn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, features: int, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

    
class SublayerConnection(nn.Module):
    "Construct a sublayer connection with layer pre-norm applied"

    def __init__(self, d_model: int, dropout = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, layer: nn.Module):
        x_norm = self.layer_norm(x)
        output_x = layer(x_norm)
        return x + self.dropout(output_x)

class Generator(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.vocab_map = nn.Linear(d_model, vocab_size)

    def forward(self, input: Tensor):
        return torch.log_softmax(self.vocab_map(input), -1)
