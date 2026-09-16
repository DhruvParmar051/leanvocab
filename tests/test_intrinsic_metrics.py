"""Metric functions checked against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from leanvocab.intrinsic.metrics import (
    EncoderShape,
    renyi_efficiency,
    renyi_entropy,
    split_paragraphs,
    tokens_per_word,
    truncated_share,
    word_spans,
)


def test_renyi_uniform_equals_log_support() -> None:
    counts = np.array([5, 5, 5, 5, 0])
    for alpha in (0.5, 1.0, 2.5, 3.0):
        assert renyi_entropy(counts, alpha) == pytest.approx(2.0)
    # uniform over 4 of an 8-token vocab: 2 / log2(8)
    assert renyi_efficiency(counts, 2.5, 8) == pytest.approx(2.0 / 3.0)


def test_renyi_hand_computed() -> None:
    # p = (0.5, 0.25, 0.25); sum p^2.5 = 0.5^2.5 + 2 * 0.25^2.5
    s = 0.5**2.5 + 2 * 0.25**2.5
    expected = math.log2(s) / (1 - 2.5)
    assert renyi_entropy(np.array([2, 1, 1]), 2.5) == pytest.approx(expected)
    assert renyi_entropy(np.array([2, 1, 1]), 1.0) == pytest.approx(1.5)
    with pytest.raises(ValueError):
        renyi_entropy(np.array([0, 0]), 2.5)


def test_renyi_matches_tokenization_scorer() -> None:
    tokenization_scorer = pytest.importorskip("tokenization_scorer")
    rng = np.random.default_rng(0)
    ids = rng.zipf(1.3, size=5000) % 300
    lines = [" ".join(f"t{i}" for i in ids[k : k + 50]) for k in range(0, ids.size, 50)]
    counts = np.bincount(ids, minlength=300)
    observed = int((counts > 0).sum())
    ours_observed = renyi_efficiency(counts, 2.5, observed)
    ours_vocab = renyi_efficiency(counts, 2.5, 1000)
    theirs_observed = tokenization_scorer.score("\n".join(lines), metric="renyi", power=2.5)
    theirs_vocab = tokenization_scorer.score(
        "\n".join(lines), metric="renyi", power=2.5, vocab=1000
    )
    assert ours_observed == pytest.approx(theirs_observed)
    assert ours_vocab == pytest.approx(theirs_vocab)


def test_word_spans_and_token_assignment() -> None:
    text = "ab  cde f"
    starts, ends = word_spans(text)
    assert starts.tolist() == [0, 4, 8] and ends.tolist() == [2, 7, 9]
    # tokens: "ab" | " " (whitespace only) | " c" (last char inside cde) | "de" | " " | "f"
    offsets = np.array([[0, 2], [2, 3], [3, 5], [5, 7], [7, 8], [8, 9]])
    assigned, per_word = tokens_per_word(offsets, starts, ends)
    assert assigned.tolist() == [0, -1, 1, 1, -1, 2]
    assert per_word.tolist() == [1, 2, 1]


def test_token_assignment_byte_fallback_and_empty() -> None:
    starts, ends = word_spans("é x")
    # three byte tokens all mapped to the single character "é"
    offsets = np.array([[0, 1], [0, 1], [0, 1], [1, 3]])
    assigned, per_word = tokens_per_word(offsets, starts, ends)
    assert per_word.tolist() == [3, 1]
    s, e = word_spans("   ")
    assert s.size == 0
    a, pw = tokens_per_word(np.zeros((0, 2), dtype=np.int64), s, e)
    assert a.size == 0 and pw.size == 0


def test_truncated_share() -> None:
    lengths = np.array([100, 126, 127, 300])
    # +2 specials: 102, 128, 129, 302 -> two exceed 128
    assert truncated_share(lengths, 128, 2) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        truncated_share(np.array([]), 128, 2)


def test_encoder_shape_params_match_minilm() -> None:
    shape = EncoderShape(n_layers=6, d_model=384, d_ffn=1536, max_len=256, type_vocab=2)
    assert shape.layer_params == 1_774_464
    assert shape.non_embedding_params == 10_646_784
    assert shape.total_params(30_522) == 22_467_072
    # all-MiniLM-L6-v2 has 512 positions and a 384x384 pooler: 22,713,216 parameters
    minilm = EncoderShape(n_layers=6, d_model=384, d_ffn=1536, max_len=512, type_vocab=2)
    assert minilm.total_params(30_522) + 384 * 384 + 384 == 22_713_216


def test_forward_flops() -> None:
    shape = EncoderShape(n_layers=1, d_model=2, d_ffn=4, max_len=8, type_vocab=2)
    n = shape.non_embedding_params
    got = shape.forward_flops(np.array([3]))
    assert got.tolist() == [2 * n * 3 + 2 * 1 * 9 * 2]


def test_split_paragraphs() -> None:
    assert split_paragraphs("  one line \n\n x\nanother one\n", 3) == ["one line", "another one"]
