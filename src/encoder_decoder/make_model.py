import copy

from torch import nn

from .decoder import Decoder, DecoderLayer
from .embeddings import Embeddings
from .encoder import Encoder, EncoderLayer
from .encoder_decoder import EncoderDecoder
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadedAttention
from .utils import Generator


def make_model(src_vocab_size, tgt_vocab_size, max_context=1024, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
  attn = MultiHeadedAttention(h, d_model, max_context, dropout)
  cross_attn = copy.deepcopy(attn)
  ff = FeedForward(d_model, d_ff, dropout)
  model = EncoderDecoder(
    Encoder(EncoderLayer(d_model, attn, ff, dropout), N, d_model),
    Decoder(DecoderLayer(d_model, attn, cross_attn, ff, dropout), N, d_model),
    Generator(d_model, tgt_vocab_size),
    Embeddings(d_model, src_vocab_size),
    Embeddings(d_model, tgt_vocab_size)
  )
  
  for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

  return model