from torch import Tensor, nn

from .utils import LayerNorm, SublayerConnection, clones


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, self_attn: nn.Module, cross_attn: nn.Module, ff: nn.Module, dropout = 0.1):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ff = ff
        self.sublayers = nn.ModuleList([SublayerConnection(d_model, dropout), SublayerConnection(d_model, dropout), SublayerConnection(d_model, dropout)])

    def forward(self, input: Tensor, encoder_output: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        input = self.sublayers[0](input, lambda t: self.self_attn(t, t, t, tgt_mask))
        input = self.sublayers[1](input, lambda t: self.cross_attn(t, encoder_output, encoder_output, src_mask))
        input = self.sublayers[2](input, self.ff)
        return input

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, decoder_layer: nn.Module, N: int, d_model: int):
        super().__init__()
        self.layers = clones(decoder_layer, N)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, input: Tensor, encoder_output: Tensor, src_mask, tgt_mask):
        for layer in self.layers:
            input = layer(input, encoder_output, src_mask, tgt_mask)
        return self.layer_norm(input)