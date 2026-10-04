import torch
from torch import nn

from .multi_head_attention import MultiHeadedAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_expand: int, dropout = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.a_1 = nn.Linear(d_model, d_expand)
        self.a_2 = nn.Linear(d_expand, d_model)

    def forward(self, x):
        return self.a_2(self.dropout(self.gelu(self.a_1(x))))
    
def main() -> None:

    torch.manual_seed(0)
    d_k = 2
    y_pos = torch.arange(3)
    freqs = 10000 ** (-2 * torch.arange(d_k // 2, dtype=torch.float) / d_k)

    angles = torch.outer(y_pos, freqs)

    cos_table = torch.cos(angles)
    sin_table = torch.cos(angles)


    # print(cos_table, sin_table)

    # Input: 1 batch, 3 tokens, 4 features each
    x = torch.tensor([[[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0], [1.0, 1.0, 1.0, 1.0]]])

    seq = x.shape[2]

    cos = cos_table[:seq]
    sin = sin_table[:seq]

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    cos = cos.repeat(1,1,1,2)
    sin = sin.repeat(1,1,1,2)
    

    mha = MultiHeadedAttention(2,2 * d_k,10, 0)
    print(mha.forward(x,x,x, None))

    