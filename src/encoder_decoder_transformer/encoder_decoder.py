"""Full encoder-decoder transformer model."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

if TYPE_CHECKING:
    from .decoder import Decoder
    from .embeddings import Embeddings
    from .encoder import Encoder
    from .utils import Generator


class EncoderDecoder(nn.Module):
    """Combine an encoder, decoder, generator and token embeddings.

    Args:
        encoder: The encoder stack.
        decoder: The decoder stack.
        generator: Output projection producing vocab log-probabilities.
        src_emb: Source token embeddings.
        tgt_emb: Target token embeddings.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        generator: Generator,
        src_emb: Embeddings,
        tgt_emb: Embeddings,
    ) -> None:
        """Store the encoder, decoder, generator and embeddings.

        Args:
            encoder: The encoder stack.
            decoder: The decoder stack.
            generator: Output projection producing vocab log-probabilities.
            src_emb: Source token embeddings.
            tgt_emb: Target token embeddings.
        """
        super().__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.generator: Generator = generator
        self.src_emb: Embeddings = src_emb
        self.tgt_emb: Embeddings = tgt_emb

    def forward(
        self,
        src: Tensor,
        target: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        """Encode ``src`` and decode ``target`` into vocab log-probabilities.

        Args:
            src: Source token indices of shape ``(batch, src_seq)``.
            target: Target token indices of shape ``(batch, tgt_seq)``.
            src_mask: Source attention mask.
            tgt_mask: Target attention mask.

        Returns:
            Log-probabilities of shape ``(batch, tgt_seq, tgt_vocab_size)``.
        """
        memory = self.encode(src, src_mask)
        decoder_output = self.decode(target, memory, src_mask, tgt_mask)
        return cast("Tensor", self.generator(decoder_output))

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """Embed and encode the source tokens.

        Args:
            src: Source token indices of shape ``(batch, src_seq)``.
            src_mask: Source attention mask.

        Returns:
            Encoder memory of shape ``(batch, src_seq, d_model)``.
        """
        return cast("Tensor", self.encoder(self.src_emb(src), src_mask))

    def decode(
        self,
        tgt: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        """Embed and decode the target tokens given the encoder output.

        Args:
            tgt: Target token indices of shape ``(batch, tgt_seq)``.
            encoder_output: Encoder memory of shape
                ``(batch, src_seq, d_model)``.
            src_mask: Source attention mask.
            tgt_mask: Target attention mask.

        Returns:
            Decoder features of shape ``(batch, tgt_seq, d_model)``.
        """
        return cast("Tensor", self.decoder(self.tgt_emb(tgt), encoder_output, src_mask, tgt_mask))
