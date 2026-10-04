"""Multi-head attention with rotary position embeddings (RoPE)."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

from .utils import clones


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    dropout: nn.Dropout | None,
) -> tuple[Tensor, Tensor]:
    """Compute scaled dot-product attention.

    Args:
        query: Query tensor of shape ``(batch, heads, seq_q, d_k)``.
        key: Key tensor of shape ``(batch, heads, seq_k, d_k)``.
        value: Value tensor of shape ``(batch, heads, seq_k, d_v)``.
        mask: Optional mask broadcastable to ``(batch, heads, seq_q, seq_k)``.
            Positions where the mask equals zero are ignored.
        dropout: Optional dropout applied to the attention weights.

    Returns:
        A tuple ``(output, attn)`` where ``output`` has shape
        ``(batch, heads, seq_q, d_v)`` and ``attn`` has shape
        ``(batch, heads, seq_q, seq_k)``.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """Multi-head self/cross attention using rotary position embeddings.

    Args:
        h: Number of attention heads.
        d_model: Feature dimension of the input.
        max_context: Maximum sequence length supported by the RoPE tables.
        dropout: Dropout probability applied to the attention weights.
    """

    def __init__(
        self,
        h: int,
        d_model: int,
        max_context: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize projections, dropout, and the RoPE buffers.

        Args:
            h: Number of attention heads.
            d_model: Feature dimension of the input.
            max_context: Maximum sequence length supported by the RoPE tables.
            dropout: Dropout probability applied to the attention weights.

        Raises:
            AssertionError: If ``d_model`` is not divisible by ``h``.
        """
        super().__init__()
        assert d_model % h == 0
        self.d_k: int = d_model // h
        self.h: int = h
        self.linears: nn.ModuleList = clones(nn.Linear(d_model, d_model), 4)
        self.attn: Tensor | None = None
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

        y_pos = torch.arange(max_context)
        freqs = 10000 ** (-2 * torch.arange(self.d_k // 2, dtype=torch.float) / self.d_k)
        angles = torch.outer(y_pos, freqs)
        self.cos_table: Tensor
        self.sin_table: Tensor
        self.register_buffer("cos_table", torch.cos(angles))
        self.register_buffer("sin_table", torch.sin(angles))

    def rope_rotate(self, tensor: Tensor) -> Tensor:
        """Apply rotary position embeddings to ``tensor``.

        Args:
            tensor: Tensor of shape ``(batch, heads, seq, d_k)``.

        Returns:
            The rotated tensor with the same shape as ``tensor``.
        """
        input_len = tensor.shape[2]

        cos = self.cos_table[:input_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_table[:input_len].unsqueeze(0).unsqueeze(0)

        first_half = tensor[..., : self.d_k // 2]
        second_half = tensor[..., self.d_k // 2 :]

        a = first_half * cos - second_half * sin
        b = first_half * sin + second_half * cos
        return torch.concat((a, b), -1)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        """Apply multi-head attention to ``query``, ``key`` and ``value``.

        Args:
            query: Query tensor of shape ``(batch, seq_q, d_model)``.
            key: Key tensor of shape ``(batch, seq_k, d_model)``.
            value: Value tensor of shape ``(batch, seq_k, d_model)``.
            mask: Optional mask broadcastable to
                ``(batch, 1, seq_q, seq_k)``. Positions where the mask equals
                zero are ignored.

        Returns:
            The attention output of shape ``(batch, seq_q, d_model)``.
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value), strict=False)
        ]

        rotated_query = self.rope_rotate(query)
        rotated_key = self.rope_rotate(key)

        x, self.attn = attention(rotated_query, rotated_key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return cast("Tensor", self.linears[-1](x))
