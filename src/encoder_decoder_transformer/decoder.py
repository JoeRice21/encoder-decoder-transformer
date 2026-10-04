"""Transformer decoder layers and stack."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

from .utils import LayerNorm, SublayerConnection, clones

if TYPE_CHECKING:
    from .feed_forward import FeedForward
    from .multi_head_attention import MultiHeadedAttention


class DecoderLayer(nn.Module):
    """A single decoder layer with self-attention, cross-attention and FFN.

    Args:
        d_model: Feature dimension of the input.
        self_attn: Multi-head self-attention module.
        cross_attn: Multi-head cross-attention module.
        ff: Position-wise feed-forward network.
        dropout: Dropout probability used inside the sublayer connections.
    """

    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadedAttention,
        cross_attn: MultiHeadedAttention,
        ff: FeedForward,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the attention and feed-forward sublayers.

        Args:
            d_model: Feature dimension of the input.
            self_attn: Multi-head self-attention module.
            cross_attn: Multi-head cross-attention module.
            ff: Position-wise feed-forward network.
            dropout: Dropout probability used inside the sublayer connections.
        """
        super().__init__()
        self.self_attn: MultiHeadedAttention = self_attn
        self.cross_attn: MultiHeadedAttention = cross_attn
        self.ff: FeedForward = ff
        self.sublayers: nn.ModuleList = nn.ModuleList(
            [
                SublayerConnection(d_model, dropout),
                SublayerConnection(d_model, dropout),
                SublayerConnection(d_model, dropout),
            ]
        )

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        """Apply self-attention, cross-attention and the feed-forward network.

        Args:
            x: Target input tensor of shape ``(batch, tgt_seq, d_model)``.
            encoder_output: Encoder output of shape
                ``(batch, src_seq, d_model)``.
            src_mask: Source attention mask broadcastable to
                ``(batch, 1, tgt_seq, src_seq)``.
            tgt_mask: Target attention mask broadcastable to
                ``(batch, 1, tgt_seq, tgt_seq)``.

        Returns:
            A tensor of shape ``(batch, tgt_seq, d_model)``.
        """
        x = self.sublayers[0](x, lambda t: self.self_attn(t, t, t, tgt_mask))
        x = self.sublayers[1](
            x, lambda t: self.cross_attn(t, encoder_output, encoder_output, src_mask)
        )
        x = self.sublayers[2](x, self.ff)
        return x


class Decoder(nn.Module):
    """Stack of ``N`` decoder layers followed by layer normalization.

    Args:
        decoder_layer: The decoder layer to clone.
        n: Number of decoder layers in the stack.
        d_model: Feature dimension of the input.
    """

    def __init__(self, decoder_layer: DecoderLayer, n: int, d_model: int) -> None:
        """Initialize the cloned layers and final layer norm.

        Args:
            decoder_layer: The decoder layer to clone.
            n: Number of decoder layers in the stack.
            d_model: Feature dimension of the input.
        """
        super().__init__()
        self.layers: nn.ModuleList = clones(decoder_layer, n)
        self.layer_norm: LayerNorm = LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        """Run ``x`` through every decoder layer.

        Args:
            x: Target input tensor of shape ``(batch, tgt_seq, d_model)``.
            encoder_output: Encoder output of shape
                ``(batch, src_seq, d_model)``.
            src_mask: Source attention mask broadcastable to
                ``(batch, 1, tgt_seq, src_seq)``.
            tgt_mask: Target attention mask broadcastable to
                ``(batch, 1, tgt_seq, tgt_seq)``.

        Returns:
            A tensor of shape ``(batch, tgt_seq, d_model)``.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return cast("Tensor", self.layer_norm(x))
