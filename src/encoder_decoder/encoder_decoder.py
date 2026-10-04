from torch import Tensor, nn


class EncoderDecoder(nn.Module):
    "Construct a sublayer connection with layer pre-norm applied"

    def __init__(self, encoder: nn.Module, decoder: nn.Module, generator: nn.Module, src_emb: nn.Module, tgt_emb: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb

    def forward(self, input, target, src_mask, tgt_mask):
       memory = self.encode(input, src_mask)
       decoder_output = self.decode(target, memory, src_mask, tgt_mask)
       return self.generator(decoder_output)

    def encode(self, input: Tensor, src_mask: Tensor):
        return self.encoder.forward(self.src_emb(input), src_mask)

    def decode(self, input: Tensor, encoder_output: Tensor, src_mask: Tensor, tgt_mask):
        return self.decoder.forward(self.tgt_emb(input), encoder_output, src_mask, tgt_mask)