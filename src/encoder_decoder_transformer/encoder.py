"""Transformer encoder layers and stack."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from .utils import LayerNorm, SublayerConnection, clones

if TYPE_CHECKING:
    from .feed_forward import FeedForward
    from .multi_head_attention import MultiHeadedAttention


class EncoderLayer(nn.Module):
    """A single encoder layer with self-attention and a feed-forward network.

    Args:
        d_model: Feature dimension of the input.
        self_attn: Multi-head self-attention module.
        ff: Position-wise feed-forward network.
        dropout: Dropout probability used inside the sublayer connections.
    """

    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadedAttention,
        ff: FeedForward,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the attention and feed-forward sublayers.

        Args:
            d_model: Feature dimension of the input.
            self_attn: Multi-head self-attention module.
            ff: Position-wise feed-forward network.
            dropout: Dropout probability used inside the sublayer connections.
        """
        super().__init__()
        self.self_attn: MultiHeadedAttention = self_attn
        self.ff: FeedForward = ff
        self.sublayers: nn.ModuleList = nn.ModuleList(
            [
                SublayerConnection(d_model, dropout),
                SublayerConnection(d_model, dropout),
            ]
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Apply self-attention then the feed-forward network.

        Args:
            x: Input tensor of shape ``(batch, seq, d_model)``.
            mask: Attention mask broadcastable to
                ``(batch, 1, seq, seq)``.

        Returns:
            A tensor of shape ``(batch, seq, d_model)``.
        """
        x = self.sublayers[0](x, lambda t: self.self_attn(t, t, t, mask))
        x = self.sublayers[1](x, self.ff)
        return x


class Encoder(nn.Module):
    """Stack of ``N`` encoder layers followed by layer normalization.

    Args:
        encoder_layer: The encoder layer to clone.
        n: Number of encoder layers in the stack.
        d_model: Feature dimension of the input.
    """

    def __init__(self, encoder_layer: EncoderLayer, n: int, d_model: int) -> None:
        """Initialize the cloned layers and final layer norm.

        Args:
            encoder_layer: The encoder layer to clone.
            n: Number of encoder layers in the stack.
            d_model: Feature dimension of the input.
        """
        super().__init__()
        self.layers: nn.ModuleList = clones(encoder_layer, n)
        self.layer_norm: LayerNorm = LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Run ``x`` through every encoder layer.

        Args:
            x: Input tensor of shape ``(batch, seq, d_model)``.
            mask: Attention mask broadcastable to
                ``(batch, 1, seq, seq)``.

        Returns:
            A tensor of shape ``(batch, seq, d_model)``.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return cast("Tensor", self.layer_norm(x))
