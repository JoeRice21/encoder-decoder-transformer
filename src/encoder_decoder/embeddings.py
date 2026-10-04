import math

from torch import Tensor, nn


class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, d_model)
            self.d_model = d_model

    def forward(self, input: Tensor):
          return self.embeddings(input) * math.sqrt(self.d_model)