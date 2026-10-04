"""Factory for building a complete encoder-decoder transformer."""

from __future__ import annotations

import copy

from torch import nn

from .decoder import Decoder, DecoderLayer
from .embeddings import Embeddings
from .encoder import Encoder, EncoderLayer
from .encoder_decoder import EncoderDecoder
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadedAttention
from .utils import Generator


def make_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    max_context: int = 1024,
    N: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
) -> EncoderDecoder:
    """Build an encoder-decoder transformer with Xavier initialization.

    Args:
        src_vocab_size: Size of the source vocabulary.
        tgt_vocab_size: Size of the target vocabulary.
        max_context: Maximum sequence length for the RoPE tables.
        N: Number of encoder and decoder layers.
        d_model: Feature dimension of the model.
        d_ff: Hidden dimension of the feed-forward network.
        h: Number of attention heads.
        dropout: Dropout probability used throughout the model.

    Returns:
        An initialized :class:`EncoderDecoder` model.
    """
    attn = MultiHeadedAttention(h, d_model, max_context, dropout)
    cross_attn = copy.deepcopy(attn)
    ff = FeedForward(d_model, d_ff, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, attn, ff, dropout), N, d_model),
        Decoder(DecoderLayer(d_model, attn, cross_attn, ff, dropout), N, d_model),
        Generator(d_model, tgt_vocab_size),
        Embeddings(d_model, src_vocab_size),
        Embeddings(d_model, tgt_vocab_size),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
