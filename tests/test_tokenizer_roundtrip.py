"""Round-trip and structure tests for the Phase 2 tokenizer builders (synthetic corpus)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from tokstudy.tokenizers.common import (
    Phase2Config,
    attach_post_processor,
    check_specials,
    load_tokenizer,
)
from tokstudy.tokenizers.roundtrip import roundtrip_report
from tokstudy.tokenizers.train_bpe import train_bpe
from tokstudy.tokenizers.train_unigram import train_unigram
from tokstudy.tokenizers.train_wordpiece import train_wordpiece
from tokstudy.tokenizers.trim_bert import select_trimmed_vocab
from tokstudy.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts" / "tokenizers"
VOCAB = 600

TRICKY = [
    "Hello, World! Don't stop.",
    "  leading and trailing spaces  ",
    "multiple   spaces\tand\ttabs\nand newlines\r\n",
    "Café naïve résumé Ångström",
    "emoji 😀🎉 and symbols ∑∫√ ©®™",
    "CJK 中文字符 日本語 한국어",
    "literal [SEP] and [MASK] tokens [CLS]",
    "numbers 3.14159 1,000,000 2026-09-16",
    "url https://example.com/a?b=c&d=e#frag",
    "",
    "zero​width and  nbsp",
    "ﬁ ligature and ｆｕｌｌｗｉｄｔｈ",
]


@pytest.fixture(scope="module")
def cfg() -> Phase2Config:
    return load_config(REPO / "configs/tokenizer/phase2.yaml", Phase2Config)


def _corpus(n_docs: int = 400) -> list[str]:
    rng = random.Random(0)
    letters = "abcdefghijklmnopqrstuvwxyz"
    words = ["".join(rng.choices(letters, k=rng.randint(2, 9))) for _ in range(3000)]
    docs = []
    for _ in range(n_docs):
        body = " ".join(rng.choice(words) for _ in range(rng.randint(20, 80)))
        docs.append(body.capitalize() + ". Numbers 123, 4567! Émigré café.")
    return docs


def _batches(docs: list[str], size: int = 50):
    for i in range(0, len(docs), size):
        yield docs[i : i + size]


@pytest.fixture(scope="module")
def trained(cfg: Phase2Config) -> dict[str, Tokenizer]:
    docs = _corpus()
    toks = {
        "bpe": train_bpe(cfg, cfg.bpe, VOCAB, _batches(docs), len(docs)),
        "uni": train_unigram(cfg, cfg.unigram, VOCAB, _batches(docs), len(docs)),
        "wp": train_wordpiece(cfg, cfg.wordpiece, VOCAB, _batches(docs), len(docs)),
    }
    for tok in toks.values():
        attach_post_processor(tok, cfg)
        tok.encode_special_tokens = True
    return toks


@pytest.mark.parametrize("name", ["bpe", "uni", "wp"])
def test_vocab_size_and_specials(
    trained: dict[str, Tokenizer], cfg: Phase2Config, name: str
) -> None:
    tok = trained[name]
    assert tok.get_vocab_size() == VOCAB
    check_specials(tok, cfg)
    enc = tok.encode("hello")
    assert enc.tokens[0] == "[CLS]" and enc.tokens[-1] == "[SEP]"


@pytest.mark.parametrize("name", ["bpe", "uni"])
def test_exact_roundtrip_on_unseen_text(
    trained: dict[str, Tokenizer], cfg: Phase2Config, name: str
) -> None:
    tok = trained[name]
    report = roundtrip_report(tok, TRICKY, "exact", cfg.unk_token)
    assert report["passed"], report["mismatch_examples"]
    unk = tok.token_to_id(cfg.unk_token)
    for text in TRICKY:
        assert unk not in tok.encode(text, add_special_tokens=False).ids


def test_unigram_uses_byte_fallback(trained: dict[str, Tokenizer]) -> None:
    tokens = trained["uni"].encode("😀", add_special_tokens=False).tokens
    assert tokens and all(t.startswith("<0x") or t == "▁" for t in tokens)


def test_wordpiece_roundtrip_ignoring_whitespace(
    trained: dict[str, Tokenizer], cfg: Phase2Config
) -> None:
    report = roundtrip_report(trained["wp"], TRICKY, "ignore_whitespace", cfg.unk_token)
    assert report["passed"], report["mismatch_examples"]


def test_literal_special_text_is_not_special(trained: dict[str, Tokenizer]) -> None:
    ids = trained["bpe"].encode("a [SEP] b", add_special_tokens=False).ids
    assert 3 not in ids  # [SEP] id


def test_roundtrip_detects_corruption(trained: dict[str, Tokenizer], cfg: Phase2Config) -> None:
    tok = trained["bpe"]

    class Lossy:
        def __getattr__(self, name):  # delegate everything except decode_batch
            return getattr(tok, name)

        def decode_batch(self, ids, skip_special_tokens=False):
            return [s + "x" for s in tok.decode_batch(ids, skip_special_tokens=skip_special_tokens)]

    report = roundtrip_report(Lossy(), ["abc"], "exact", cfg.unk_token)  # type: ignore[arg-type]
    assert not report["passed"] and report["mismatched"] == 1


def test_trim_keeps_specials_chars_then_frequency() -> None:
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tokens = specials + ["a", "b", "c", "##a", "##b", "##c", "ab", "##ca", "cab", "zz"]
    vocab = {t: i for i, t in enumerate(tokens)}
    counts = np.zeros(len(tokens), dtype=np.int64)
    counts[vocab["ab"]] = 10
    counts[vocab["##ca"]] = 5
    counts[vocab["cab"]] = 7
    counts[vocab["[UNK]"]] = 99
    # chars used: a, b, c -> 6 char tokens; budget = 12 - 5 - 6 = 1 -> "ab" (most frequent)
    ordered, stats = select_trimmed_vocab(vocab, counts, specials, 12)
    assert ordered[:5] == specials
    assert set(ordered[5:]) == {"a", "b", "c", "##a", "##b", "##c", "ab"}
    assert ordered[5:] == sorted(ordered[5:], key=vocab.__getitem__)
    assert stats["char_tokens_kept"] == 6 and stats["frequency_tokens_kept"] == 1
    with pytest.raises(ValueError):
        select_trimmed_vocab(vocab, counts, specials, 10)


@pytest.mark.skipif(not ARTIFACTS.is_dir(), reason="Phase 2 artifacts not built")
@pytest.mark.parametrize(
    "tok_dir", sorted(p for p in ARTIFACTS.glob("*") if p.is_dir()) if ARTIFACTS.is_dir() else []
)
def test_saved_tokenizers_roundtrip(tok_dir: Path, cfg: Phase2Config) -> None:
    tok, meta = load_tokenizer(tok_dir)
    assert meta["roundtrip"]["passed"]
    mode = meta["roundtrip"]["mode"]
    report = roundtrip_report(tok, TRICKY, mode, cfg.unk_token)
    assert report["passed"], report["mismatch_examples"]
    assert tok.get_vocab_size() == meta["vocab_size"]
