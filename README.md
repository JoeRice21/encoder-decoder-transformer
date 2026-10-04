# encoder-decoder-transformer

A from-scratch PyTorch implementation of the Transformer encoder-decoder
architecture, including multi-head attention with rotary position embeddings
(RoPE), pre-layer normalization, and a position-wise feed-forward network.

## Installation

Install from source with [uv](https://docs.astral.sh/uv/):

```bash
uv add encoder-decoder-transformer
```

or with pip:

```bash
pip install encoder-decoder-transformer
```

The package targets Python 3.13 and depends on PyTorch.

## Usage

Build a model and run a forward pass. Token indices are integer tensors of
shape `(batch, seq)`; the model returns log-probabilities of shape
`(batch, tgt_seq, tgt_vocab_size)`. Attention masks are boolean/integer tensors
of shape `(batch, seq_q, seq_k)` (the model adds the head dimension).

```python
import torch

from encoder_decoder import make_model

model = make_model(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    max_context=128,
    N=2,
    d_model=64,
    d_ff=256,
    h=4,
    dropout=0.1,
)

src = torch.randint(0, 1000, (2, 8))
tgt = torch.randint(0, 1000, (2, 8))
src_mask = torch.ones(2, 8, 8)
tgt_mask = torch.ones(2, 8, 8)

log_probs = model(src, tgt, src_mask, tgt_mask)
print(log_probs.shape)  # torch.Size([2, 8, 1000])
```

## Development

Run the linters and type checker with:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

## License

MIT
