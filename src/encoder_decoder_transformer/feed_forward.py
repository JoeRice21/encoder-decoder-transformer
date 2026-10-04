"""Position-wise feed-forward network."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn


class FeedForward(nn.Module):
    """Two-layer position-wise feed-forward network with GELU activation.

    Args:
        d_model: Input and output feature dimension.
        d_expand: Hidden (expanded) feature dimension.
        dropout: Dropout probability applied between the linear layers.
    """

    def __init__(self, d_model: int, d_expand: int, dropout: float = 0.1) -> None:
        """Initialize the linear layers and activation.

        Args:
            d_model: Input and output feature dimension.
            d_expand: Hidden (expanded) feature dimension.
            dropout: Dropout probability applied between the linear layers.
        """
        super().__init__()
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.gelu: nn.GELU = nn.GELU()
        self.a_1: nn.Linear = nn.Linear(d_model, d_expand)
        self.a_2: nn.Linear = nn.Linear(d_expand, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the feed-forward network to ``x``.

        Args:
            x: Input tensor of shape ``(batch, seq, d_model)``.

        Returns:
            A tensor of shape ``(batch, seq, d_model)``.
        """
        return cast("Tensor", self.a_2(self.dropout(self.gelu(self.a_1(x)))))
