"""Shared character alphabet for WordPiece and Unigram, and the filtered Unigram training text.

The 5 GB subset has ~11k distinct characters after BERT-uncased normalization, more than an
8k vocab can hold. Both character-level families therefore use the same top-``size``
characters (count desc, code point asc; deterministic):

- WordPiece gets them as ``initial_alphabet``; other characters become [UNK].
- Unigram (HF requires every training character in the vocab) trains on normalized text in
  which other characters are replaced by a space, i.e. a word boundary, like SentencePiece's
  ``character_coverage``. At inference they are covered by byte fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from leanvocab.tokenizers.common import NormalizerConfig, build_normalizer

logger = logging.getLogger(__name__)

ALPHABET_FILE = "char_alphabet.json"
CACHE_DONE_FILE = "done.json"
RARE_CHAR_REPLACEMENT = " "


@dataclass(frozen=True)
class CharAlphabet:
    chars: list[str]
    coverage: float
    distinct: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256("".join(self.chars).encode("utf-8")).hexdigest()


def rare_char_pattern(chars: list[str]) -> re.Pattern[str]:
    """Matches any character that is neither in ``chars`` nor whitespace."""
    return re.compile("[^" + re.escape("".join(sorted(chars))) + r"\s]")


def _count_shard_chars(args: tuple[Path, int, NormalizerConfig]) -> Counter[str]:
    path, n_docs, norm_cfg = args
    normalizer = build_normalizer(norm_cfg)
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()[:n_docs]
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(normalizer.normalize_str(text))
    return counts


def rank_chars(
    data_dir: Path, subset: dict[str, Any], norm_cfg: NormalizerConfig
) -> list[tuple[str, int]]:
    """All non-whitespace characters of the normalized subset, by (count desc, code point)."""
    jobs = [(data_dir / s["path"], int(s["docs"]), norm_cfg) for s in subset["shards"]]
    total: Counter[str] = Counter()
    with ProcessPoolExecutor() as pool:
        for counts in pool.map(_count_shard_chars, jobs):
            total.update(counts)
    return sorted(
        ((c, n) for c, n in total.items() if not c.isspace()), key=lambda cn: (-cn[1], ord(cn[0]))
    )


def select_alphabet(ranked: list[tuple[str, int]], size: int) -> CharAlphabet:
    total = sum(n for _, n in ranked)
    kept = ranked[:size]
    return CharAlphabet(
        chars=[c for c, _ in kept],
        coverage=sum(n for _, n in kept) / total if total else 0.0,
        distinct=len(ranked),
    )


def load_or_compute_alphabet(
    out_dir: Path,
    data_dir: Path,
    subset: dict[str, Any],
    norm_cfg: NormalizerConfig,
    size: int,
) -> CharAlphabet:
    path = out_dir / ALPHABET_FILE
    if path.exists():
        stored = json.loads(path.read_text())
        if stored["subset_doc_ids_sha256"] != subset["doc_ids_sha256"]:
            raise ValueError(f"{path} was computed on a different subset")
        ranked = [(c, int(n)) for c, n in stored["ranked"]]
    else:
        ranked = rank_chars(data_dir, subset, norm_cfg)
        payload = {"subset_doc_ids_sha256": subset["doc_ids_sha256"], "ranked": ranked}
        path.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    alphabet = select_alphabet(ranked, size)
    logger.info(
        "Alphabet: %d distinct chars, keeping %d (%.5f%% of occurrences)",
        alphabet.distinct,
        len(alphabet.chars),
        100 * alphabet.coverage,
    )
    return alphabet


def _filter_shard(args: tuple[Path, int, Path, NormalizerConfig, list[str]]) -> int:
    src, n_docs, dst, norm_cfg, chars = args
    normalizer = build_normalizer(norm_cfg)
    pattern = rare_char_pattern(chars)
    texts = pq.read_table(src, columns=["text"]).column("text").to_pylist()[:n_docs]
    out = [pattern.sub(RARE_CHAR_REPLACEMENT, normalizer.normalize_str(t)) for t in texts]
    tmp = dst.with_suffix(".tmp")
    pq.write_table(pa.table({"text": out}), tmp, compression="zstd")
    tmp.replace(dst)
    return len(out)


def build_unigram_cache(
    cache_root: Path,
    data_dir: Path,
    subset: dict[str, Any],
    norm_cfg: NormalizerConfig,
    alphabet: CharAlphabet,
) -> tuple[Path, dict[str, Any]]:
    """Normalized, rare-char-filtered copy of the subset. Returns (root, subset-like spec)."""
    key = hashlib.sha256(
        json.dumps(
            [subset["doc_ids_sha256"], norm_cfg.model_dump(), alphabet.sha256], sort_keys=True
        ).encode()
    ).hexdigest()[:16]
    root = cache_root / f"unigram-{key}"
    spec = {"shards": [{"path": Path(s["path"]).name, "docs": s["docs"]} for s in subset["shards"]]}
    if (root / CACHE_DONE_FILE).exists():
        logger.info("Using cached Unigram training text %s", root)
        return root, spec
    root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (data_dir / s["path"], int(s["docs"]), root / t["path"], norm_cfg, alphabet.chars)
        for s, t in zip(subset["shards"], spec["shards"], strict=True)
    ]
    with ProcessPoolExecutor() as pool:
        written = sum(pool.map(_filter_shard, jobs))
    if written != subset["docs"]:
        raise ValueError(f"Filtered cache has {written} docs, expected {subset['docs']}")
    (root / CACHE_DONE_FILE).write_text(json.dumps({"key": key, "docs": written}) + "\n")
    logger.info("Built Unigram training text cache %s (%d docs)", root, written)
    return root, spec
