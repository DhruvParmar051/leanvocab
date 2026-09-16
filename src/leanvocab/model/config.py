"""Encoder hyperparameters (no MLX import, so entry points can configure MLX first)."""

from __future__ import annotations

from leanvocab.utils.config import StrictConfig


class EncoderConfig(StrictConfig):
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    max_position_embeddings: int
    type_vocab_size: int
    hidden_dropout: float
    layer_norm_eps: float
    initializer_range: float
    embedding_dim: int | None = None

    @property
    def factorized(self) -> bool:
        return self.embedding_dim is not None and self.embedding_dim != self.hidden_size

    @property
    def token_dim(self) -> int:
        return self.embedding_dim if self.embedding_dim is not None else self.hidden_size
