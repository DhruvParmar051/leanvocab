"""Shared pieces for every tokenizer in the study: config, normalization, corpus subset, I/O."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
from pydantic import Field
from tokenizers import Tokenizer, normalizers, processors

from tokstudy.utils.config import StrictConfig

logger = logging.getLogger(__name__)

TOKENIZER_FILE = "tokenizer.json"
META_FILE = "meta.json"
SUBSET_FILE = "train_subset.json"


class NormalizerConfig(StrictConfig):
    clean_text: bool
    handle_chinese_chars: bool
    strip_accents: bool | None
    lowercase: bool


class BpeConfig(StrictConfig):
    vocab_sizes: list[int]
    min_frequency: int = Field(ge=1)


class UnigramConfig(StrictConfig):
    vocab_sizes: list[int]
    shrinking_factor: float = Field(gt=0.0, lt=1.0)
    n_sub_iterations: int = Field(ge=1)
    max_piece_length: int = Field(ge=1)
    score_decimals: int = Field(
        ge=1, description="Scores are rounded so float noise from parallel EM cannot reorder ids"
    )


class WordPieceConfig(StrictConfig):
    vocab_sizes: list[int]
    min_frequency: int = Field(ge=1)
    limit_alphabet: int = Field(ge=1)


class BertConfig(StrictConfig):
    repo: str
    revision: str = Field(min_length=40, max_length=40)
    trim_sizes: list[int]


class Phase2Config(StrictConfig):
    seed: int = Field(ge=0)
    data_dir: Path
    subset_bytes: int = Field(gt=0)
    batch_docs: int = Field(gt=0)
    roundtrip_docs: int = Field(gt=0)
    out_dir: Path
    special_tokens: list[str]
    unk_token: str
    cls_token: str
    sep_token: str
    normalizer: NormalizerConfig
    bpe: BpeConfig
    unigram: UnigramConfig
    wordpiece: WordPieceConfig
    bert: BertConfig
    only: list[str] | None = Field(default=None, description="Restrict the run to these ids")


RoundtripMode = Literal["exact", "ignore_whitespace"]


def build_normalizer(cfg: NormalizerConfig) -> normalizers.Normalizer:
    """BERT-uncased normalization shared by every trained tokenizer (matches bert-orig)."""
    return normalizers.BertNormalizer(
        clean_text=cfg.clean_text,
        handle_chinese_chars=cfg.handle_chinese_chars,
        strip_accents=cfg.strip_accents,
        lowercase=cfg.lowercase,
    )


def attach_post_processor(tok: Tokenizer, cfg: Phase2Config) -> None:
    """Wrap single sequences as ``[CLS] x [SEP]`` and pairs as ``[CLS] a [SEP] b [SEP]``."""
    cls_id = tok.token_to_id(cfg.cls_token)
    sep_id = tok.token_to_id(cfg.sep_token)
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer is missing [CLS]/[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single=f"{cfg.cls_token} $A {cfg.sep_token}",
        pair=f"{cfg.cls_token} $A {cfg.sep_token} $B:1 {cfg.sep_token}:1",
        special_tokens=[(cfg.cls_token, cls_id), (cfg.sep_token, sep_id)],
    )


def check_specials(tok: Tokenizer, cfg: Phase2Config) -> None:
    """Special tokens must occupy ids 0..k-1 in the configured order."""
    for expected_id, token in enumerate(cfg.special_tokens):
        got = tok.token_to_id(token)
        if got != expected_id:
            raise ValueError(f"Special token {token} has id {got}, expected {expected_id}")


# --- training corpus subset -------------------------------------------------------------


def _train_shards(data_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    shards: list[dict[str, Any]] = manifest["train"]["shards"]
    return shards


def select_subset(data_dir: Path, subset_bytes: int) -> dict[str, Any]:
    """Prefix of the train shards (in manifest order) holding at least ``subset_bytes`` of text.

    Whole documents only; the document that crosses the budget is included.
    """
    fingerprint = hashlib.sha256()
    used: list[dict[str, Any]] = []
    total_bytes = total_docs = 0
    for shard in _train_shards(data_dir):
        table = pq.read_table(data_dir / shard["path"], columns=["id", "n_bytes"])
        ids = table.column("id").to_pylist()
        sizes = table.column("n_bytes").to_pylist()
        take = 0
        for doc_id, n in zip(ids, sizes, strict=True):
            if total_bytes >= subset_bytes:
                break
            fingerprint.update(doc_id.encode("utf-8") + b"\n")
            total_bytes += n
            total_docs += 1
            take += 1
        if take:
            used.append({"path": shard["path"], "sha256": shard["sha256"], "docs": take})
        if total_bytes >= subset_bytes:
            break
    if total_bytes < subset_bytes:
        raise ValueError(f"Train split has only {total_bytes} bytes, need {subset_bytes}")
    return {
        "subset_bytes_target": subset_bytes,
        "text_bytes": total_bytes,
        "docs": total_docs,
        "shards": used,
        "doc_ids_sha256": fingerprint.hexdigest(),
    }


def iter_subset_batches(
    data_dir: Path, subset: dict[str, Any], batch_docs: int
) -> Iterator[list[str]]:
    """Yield the subset's texts in fixed order, in batches of ``batch_docs``."""
    for shard in subset["shards"]:
        parquet = pq.ParquetFile(data_dir / shard["path"])
        remaining = int(shard["docs"])
        try:
            for batch in parquet.iter_batches(batch_size=batch_docs, columns=["text"]):
                texts = batch.column(0).to_pylist()[:remaining]
                remaining -= len(texts)
                if texts:
                    yield texts
                if remaining == 0:
                    break
        finally:
            parquet.close()
        if remaining != 0:
            raise ValueError(f"{shard['path']} ended {remaining} docs short of the subset spec")


def _count_shard_chars(args: tuple[Path, int, NormalizerConfig]) -> Counter[str]:
    path, n_docs, norm_cfg = args
    normalizer = build_normalizer(norm_cfg)
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()[:n_docs]
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(normalizer.normalize_str(text))
    return counts


def compute_alphabet(
    data_dir: Path, subset: dict[str, Any], norm_cfg: NormalizerConfig, limit: int
) -> list[dict[str, Any]]:
    """Top-``limit`` non-whitespace characters of the normalized subset.

    Ties are broken by code point, making the choice deterministic (HF's own
    ``limit_alphabet`` cut breaks ties in hash-map order).
    """
    jobs = [(data_dir / s["path"], int(s["docs"]), norm_cfg) for s in subset["shards"]]
    total: Counter[str] = Counter()
    with ProcessPoolExecutor() as pool:
        for counts in pool.map(_count_shard_chars, jobs):
            total.update(counts)
    ranked = sorted(
        ((c, n) for c, n in total.items() if not c.isspace()), key=lambda cn: (-cn[1], ord(cn[0]))
    )
    logger.info("Alphabet: %d distinct characters, keeping %d", len(ranked), limit)
    return [{"char": c, "count": n} for c, n in ranked[:limit]]


def iter_heldout(data_dir: Path, limit: int) -> list[str]:
    """First ``limit`` held-out documents (held-out is ordered by hash, i.e. a random sample)."""
    manifest = json.loads((data_dir / "manifest.json").read_text())
    table = pq.read_table(data_dir / manifest["heldout"]["path"], columns=["text"])
    texts: list[str] = table.column("text").to_pylist()[:limit]
    return texts


# --- persistence -----------------------------------------------------------------------


def save_tokenizer(tok: Tokenizer, out_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)
    path = out_dir / TOKENIZER_FILE
    tok.save(str(path), pretty=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    full_meta = {**meta, "vocab_size": tok.get_vocab_size(), "tokenizer_sha256": digest}
    (out_dir / META_FILE).write_text(json.dumps(full_meta, indent=2, sort_keys=True) + "\n")
    logger.info("Saved %s (vocab=%d)", out_dir, tok.get_vocab_size())
    return full_meta


def load_tokenizer(tok_dir: Path) -> tuple[Tokenizer, dict[str, Any]]:
    tok = Tokenizer.from_file(str(tok_dir / TOKENIZER_FILE))
    # Not serialized by `tokenizers`: literal "[SEP]" etc. in corpus text stays plain text.
    tok.encode_special_tokens = True
    meta: dict[str, Any] = json.loads((tok_dir / META_FILE).read_text())
    return tok, meta
