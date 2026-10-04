"""Token embedding layer."""

from __future__ import annotations

import math
from typing import cast

from torch import Tensor, nn


class Embeddings(nn.Module):
    """Embed token indices and scale them by ``sqrt(d_model)``.

    Args:
        d_model: Dimension of the embedding vectors.
        vocab_size: Number of tokens in the vocabulary.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """Initialize the embedding table.

        Args:
            d_model: Dimension of the embedding vectors.
            vocab_size: Number of tokens in the vocabulary.
        """
        super().__init__()
        self.embeddings: nn.Embedding = nn.Embedding(vocab_size, d_model)
        self.d_model: int = d_model

    def forward(self, x: Tensor) -> Tensor:
        """Embed ``x`` and scale by ``sqrt(d_model)``.

        Args:
            x: Token indices of shape ``(batch, seq)``.

        Returns:
            Embeddings of shape ``(batch, seq, d_model)``.
        """
        return cast("Tensor", self.embeddings(x) * math.sqrt(self.d_model))
