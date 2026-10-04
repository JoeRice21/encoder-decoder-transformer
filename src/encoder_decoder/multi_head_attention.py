
import math

import torch
from torch import nn

from .utils import clones


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None, dropout: nn.Dropout | None):
    "Compute 'Scaled Dot Product Attention'"

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, max_context: int, dropout=0.1):
        "Take in model size and number of heads."

        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        y_pos = torch.arange(max_context)
        freqs = 10000 ** (-2 * torch.arange(self.d_k // 2, dtype=torch.float) / self.d_k)
        angles = torch.outer(y_pos, freqs)
        self.register_buffer('cos_table', torch.cos(angles))
        self.register_buffer('sin_table', torch.sin(angles))

    def rope_rotate (self, tensor: torch.Tensor):
        input_len = tensor.shape[2]
            
        cos = self.cos_table[:input_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_table[:input_len].unsqueeze(0).unsqueeze(0)
        
        cos = cos.repeat(1, 1, 1, 2)
        sin = sin.repeat(1, 1, 1, 2)
        first_half, second_half = tensor.split(self.d_k // 2, -1)

        a = first_half * cos - second_half * sin
        b = first_half * sin + second_half * cos
        return torch.concat((a,b), -1)


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None):
        "Implements Figure 2"

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        rotated_query = self.rope_rotate(query)
        rotated_key = self.rope_rotate(key)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            rotated_query, rotated_key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)