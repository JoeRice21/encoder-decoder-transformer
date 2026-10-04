"""Reusable building blocks for the encoder-decoder transformer."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Callable


def clones[ModuleT: nn.Module](module: ModuleT, n: int) -> nn.ModuleList:
    """Create ``n`` independent deep copies of ``module``.

    Args:
        module: The module to duplicate.
        n: Number of copies to create.

    Returns:
        A ``ModuleList`` holding the ``n`` deep copies of ``module``.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension of the input.

    Args:
        features: Size of the normalized (last) dimension.
        eps: Small constant added to the standard deviation for numerical
            stability.
    """

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        """Initialize the learnable affine parameters.

        Args:
            features: Size of the normalized (last) dimension.
            eps: Small constant added to the standard deviation.
        """
        super().__init__()
        self.a_2: nn.Parameter = nn.Parameter(torch.ones(features))
        self.b_2: nn.Parameter = nn.Parameter(torch.zeros(features))
        self.eps: float = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``x`` along its last dimension.

        Args:
            x: Input tensor of shape ``(..., features)``.

        Returns:
            A tensor with the same shape as ``x``, normalized along the last
            dimension.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """Residual connection wrapped around a pre-normalized sublayer.

    Args:
        d_model: Feature dimension of the input.
        dropout: Dropout probability applied to the sublayer output.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        """Initialize the layer norm and dropout module.

        Args:
            d_model: Feature dimension of the input.
            dropout: Dropout probability applied to the sublayer output.
        """
        super().__init__()
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)
        self.layer_norm: LayerNorm = LayerNorm(d_model)

    def forward(self, x: Tensor, layer: Callable[[Tensor], Tensor]) -> Tensor:
        """Apply ``layer`` to the normalized input and add the residual.

        Args:
            x: Input tensor of shape ``(..., d_model)``.
            layer: Callable applied to the normalized input.

        Returns:
            A tensor of shape ``(..., d_model)`` produced by
            ``x + dropout(layer(layer_norm(x)))``.
        """
        x_norm = self.layer_norm(x)
        output_x = layer(x_norm)
        return cast("Tensor", x + self.dropout(output_x))


class Generator(nn.Module):
    """Map decoder features to log-probabilities over a vocabulary.

    Args:
        d_model: Feature dimension of the decoder output.
        vocab_size: Number of tokens in the target vocabulary.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """Initialize the output projection.

        Args:
            d_model: Feature dimension of the decoder output.
            vocab_size: Number of tokens in the target vocabulary.
        """
        super().__init__()
        self.vocab_map: nn.Linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        """Project ``x`` and apply a log-softmax over the vocabulary.

        Args:
            x: Decoder features of shape ``(batch, seq, d_model)``.

        Returns:
            Log-probabilities of shape ``(batch, seq, vocab_size)``.
        """
        return torch.log_softmax(self.vocab_map(x), -1)
