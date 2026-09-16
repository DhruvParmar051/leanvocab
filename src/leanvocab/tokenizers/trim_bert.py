"""bert-orig (bert-base-uncased, as used by MiniLM) and frequency-trimmed variants of it.

Trimming rule for a target size ``v``:
1. the special tokens, at ids 0..4 in the configured order;
2. ``c`` and ``##c`` for every character ``c`` that occurs in any BERT token used on the
   training subset, so words whose original pieces were dropped can still be spelled out;
3. the remaining slots filled by token frequency on the subset (ties: lower original id).
Kept tokens keep their original relative order after the specials. ``id_map`` records the
original id of every new id.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer, models

from leanvocab.tokenizers.common import Phase2Config, attach_post_processor
from leanvocab.tokenizers.train_wordpiece import CONTINUATION_PREFIX, MAX_INPUT_CHARS_PER_WORD

logger = logging.getLogger(__name__)


def load_bert_orig(cfg: Phase2Config) -> tuple[Tokenizer, Path]:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(cfg.bert.repo, "tokenizer.json", revision=cfg.bert.revision))
    tok = Tokenizer.from_file(str(path))
    for token in cfg.special_tokens:
        if tok.token_to_id(token) is None:
            raise ValueError(f"{cfg.bert.repo} lacks special token {token}")
    return tok, path


def count_token_frequencies(tok: Tokenizer, batches: Iterator[list[str]]) -> np.ndarray:
    counts = np.zeros(tok.get_vocab_size(), dtype=np.int64)
    for texts in batches:
        encodings = tok.encode_batch_fast(texts, add_special_tokens=False)
        for enc in encodings:
            counts += np.bincount(np.asarray(enc.ids, dtype=np.int64), minlength=counts.size)
    return counts


def select_trimmed_vocab(
    vocab: dict[str, int], counts: np.ndarray, specials: list[str], size: int
) -> tuple[list[str], dict[str, int]]:
    """Return the kept tokens in new-id order and selection statistics."""
    id_to_token = {i: t for t, i in vocab.items()}
    special_set = set(specials)
    nonzero = [id_to_token[int(i)] for i in np.flatnonzero(counts)]
    used = [t for t in nonzero if t not in special_set]

    chars: set[str] = set()
    for token in used:
        chars.update(token.removeprefix(CONTINUATION_PREFIX))
    char_tokens = {t for c in chars for t in (c, CONTINUATION_PREFIX + c) if t in vocab}

    budget = size - len(specials) - len(char_tokens)
    if budget < 0:
        raise ValueError(f"size {size} cannot hold {len(char_tokens)} character tokens")
    ranked = sorted(
        (t for t in used if t not in char_tokens),
        key=lambda t: (-int(counts[vocab[t]]), vocab[t]),
    )
    if len(ranked) < budget:
        raise ValueError(f"Only {len(ranked)} used non-character tokens, need {budget}")
    kept = char_tokens | set(ranked[:budget])
    ordered = specials + sorted(kept, key=lambda t: vocab[t])
    stats = {
        "used_tokens": len(used),
        "char_tokens_kept": len(char_tokens),
        "frequency_tokens_kept": budget,
        "min_kept_count": int(counts[vocab[ranked[budget - 1]]]) if budget else 0,
    }
    return ordered, stats


def build_trimmed(
    cfg: Phase2Config, orig: Tokenizer, counts: np.ndarray, size: int
) -> tuple[Tokenizer, dict[str, Any]]:
    vocab = orig.get_vocab()
    ordered, stats = select_trimmed_vocab(vocab, counts, cfg.special_tokens, size)
    new_vocab = {t: i for i, t in enumerate(ordered)}
    tok = Tokenizer(
        models.WordPiece(
            new_vocab,
            unk_token=cfg.unk_token,
            continuing_subword_prefix=CONTINUATION_PREFIX,
            max_input_chars_per_word=MAX_INPUT_CHARS_PER_WORD,
        )
    )
    tok.normalizer = orig.normalizer
    tok.pre_tokenizer = orig.pre_tokenizer
    tok.decoder = orig.decoder
    tok.add_special_tokens(cfg.special_tokens)
    attach_post_processor(tok, cfg)
    total = int(counts.sum())
    kept_mass = int(sum(counts[vocab[t]] for t in ordered))
    meta = {
        **stats,
        "orig_token_mass_covered": kept_mass / total if total else 0.0,
        "id_map": [vocab[t] for t in ordered],
    }
    return tok, meta
