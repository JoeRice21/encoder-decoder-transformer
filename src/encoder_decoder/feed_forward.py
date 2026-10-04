from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_expand: int, dropout = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.a_1 = nn.Linear(d_model, d_expand)
        self.a_2 = nn.Linear(d_expand, d_model)

    def forward(self, x):
        return self.a_2(self.dropout(self.gelu(self.a_1(x))))