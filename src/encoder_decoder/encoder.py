from torch import Tensor, nn

from .utils import LayerNorm, SublayerConnection, clones


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, self_attn: nn.Module, ff: nn.Module, dropout = 0.1):
        super().__init__()
        self.self_attn = self_attn
        self.ff = ff
        self.sublayers = nn.ModuleList([SublayerConnection(d_model, dropout), SublayerConnection(d_model, dropout)])

    def forward(self, input: Tensor, mask: Tensor):
        input = self.sublayers[0](input, lambda t: self.self_attn(t, t, t, mask))
        input = self.sublayers[1](input, self.ff)
        return input

class Encoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, N: int, d_model: int):
        super().__init__()
        self.layers = clones(encoder_layer, N)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, input: Tensor, mask: Tensor):
        for layer in self.layers:
            input = layer(input, mask)
        return self.layer_norm.forward(input)