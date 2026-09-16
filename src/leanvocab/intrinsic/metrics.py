"""Pure metric functions for Phase 3 (intrinsic tokenizer evaluation)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_WORD = re.compile(r"\S+")


def renyi_entropy(counts: np.ndarray, alpha: float) -> float:
    """Rényi entropy (bits) of the unigram distribution given by ``counts``."""
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    c = counts[counts > 0].astype(np.float64)
    if c.size == 0:
        raise ValueError("No tokens counted")
    p = c / c.sum()
    if alpha == 1.0:
        return float(-(p * np.log2(p)).sum())
    return float(np.log2((p**alpha).sum()) / (1.0 - alpha))


def renyi_efficiency(counts: np.ndarray, alpha: float, vocab_size: int) -> float:
    """H_alpha / log2(vocab_size) (Zouhar et al., 2023)."""
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    return renyi_entropy(counts, alpha) / float(np.log2(vocab_size))


def word_spans(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Start and end character offsets of whitespace-delimited words."""
    spans = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    if not spans:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    arr = np.asarray(spans, dtype=np.int64)
    return arr[:, 0], arr[:, 1]


def tokens_per_word(
    offsets: np.ndarray, word_starts: np.ndarray, word_ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each token to the word containing its last character.

    ``offsets`` is an (n, 2) array of [start, end) character offsets into the original text.
    Returns (word index per token, -1 when the token covers only whitespace; token count per
    word).
    """
    if offsets.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(word_starts.size, dtype=np.int64)
    last = np.maximum(offsets[:, 1] - 1, offsets[:, 0])
    idx = np.searchsorted(word_starts, last, side="right") - 1
    valid = idx >= 0
    valid[valid] &= last[valid] < word_ends[idx[valid]]
    assigned = np.where(valid, idx, -1)
    per_word = np.bincount(assigned[valid], minlength=word_starts.size)
    return assigned, per_word


def truncated_share(lengths: np.ndarray, max_len: int, n_special: int) -> float:
    """Share of sequences whose tokens plus ``n_special`` exceed ``max_len``."""
    if lengths.size == 0:
        raise ValueError("No sequences")
    return float(((lengths + n_special) > max_len).mean())


@dataclass(frozen=True)
class EncoderShape:
    n_layers: int
    d_model: int
    d_ffn: int
    max_len: int
    type_vocab: int

    @property
    def layer_params(self) -> int:
        d, f = self.d_model, self.d_ffn
        attention = 4 * (d * d + d)
        ffn = d * f + f + f * d + d
        layer_norms = 2 * 2 * d
        return attention + ffn + layer_norms

    @property
    def non_embedding_params(self) -> int:
        """Transformer layers only (the matmul parameters that set per-token compute)."""
        return self.n_layers * self.layer_params

    def other_embedding_params(self) -> int:
        """Position and token-type embeddings plus the embedding LayerNorm."""
        return (self.max_len + self.type_vocab) * self.d_model + 2 * self.d_model

    def token_embedding_params(self, vocab_size: int) -> int:
        return vocab_size * self.d_model

    def total_params(self, vocab_size: int) -> int:
        return (
            self.non_embedding_params
            + self.other_embedding_params()
            + self.token_embedding_params(vocab_size)
        )

    def forward_flops(self, seq_len: np.ndarray) -> np.ndarray:
        """Forward FLOPs per sequence: 2*N_layers*L + 2*n_layers*L^2*d (Kaplan et al., 2020).

        The embedding lookup is not a matmul and is ignored; ``seq_len`` should already be
        capped at ``max_len``.
        """
        seq = seq_len.astype(np.float64)
        return 2.0 * self.non_embedding_params * seq + 2.0 * self.n_layers * seq**2 * self.d_model


def split_paragraphs(text: str, min_chars: int) -> list[str]:
    return [p for p in (line.strip() for line in text.split("\n")) if len(p) >= min_chars]
