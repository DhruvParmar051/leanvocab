"""Pre-tokenized, packed MLM data with byte accounting.

Every arm reads the *same documents* (a prefix of the train shards chosen by raw bytes), so
the budget is equal in text, not in tokens. Documents are tokenized without specials and
concatenated with ``[SEP]`` between them (RoBERTa FULL-SENTENCES packing); sequence ``i`` is
``[CLS] + stream[i*(L-2) : (i+1)*(L-2)] + [SEP]``. Each document's raw bytes are spread
evenly over its tokens (and its separator), so each sequence carries the bytes of text it
covers.

For held-out bits-per-byte, tokens instead carry their own bytes (from character offsets,
rescaled so every document sums to its UTF-8 length; separators carry 0 bytes).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from leanvocab.tokenizers.common import load_tokenizer

logger = logging.getLogger(__name__)

TOKENS_FILE = "tokens.bin"
SEQ_BYTES_FILE = "seq_bytes.npy"
EVAL_TOKENS_FILE = "eval_tokens.bin"
EVAL_BYTES_FILE = "eval_token_bytes.bin"
META_FILE = "meta.json"
TOKEN_DTYPE = np.uint16
_UNUSED = re.compile(r"^\[unused\d+\]$")


@dataclass(frozen=True)
class SpecialIds:
    pad: int
    unk: int
    cls: int
    sep: int
    mask: int
    random_low: int  # random replacements are drawn from [random_low, vocab_size)
    vocab_size: int

    @property
    def unmaskable(self) -> list[int]:
        return [self.pad, self.cls, self.sep, self.mask]


def special_ids(tok: Tokenizer, specials: list[str]) -> SpecialIds:
    pad, unk, cls, sep, mask = (tok.token_to_id(t) for t in specials)
    if None in (pad, unk, cls, sep, mask):
        raise ValueError(f"Tokenizer lacks one of {specials}")
    vocab = tok.get_vocab()
    reserved = [i for t, i in vocab.items() if t in specials or _UNUSED.match(t)]
    random_low = max(reserved) + 1
    vocab_size = tok.get_vocab_size()
    if any(i >= random_low for t, i in vocab.items() if t in specials):
        raise ValueError("Special tokens must precede regular tokens")
    return SpecialIds(pad, unk, cls, sep, mask, random_low, vocab_size)  # type: ignore[arg-type]


def select_documents(data_dir: Path, budget_bytes: int) -> list[tuple[str, int]]:
    """(shard path, docs to take) for the train prefix reaching ``budget_bytes``."""
    manifest = json.loads((data_dir / "manifest.json").read_text())
    plan: list[tuple[str, int]] = []
    total = 0
    for shard in manifest["train"]["shards"]:
        sizes = pq.read_table(data_dir / shard["path"], columns=["n_bytes"]).column(0)
        cum = np.cumsum(np.asarray(sizes.to_pylist(), dtype=np.int64))
        need = budget_bytes - total
        if cum[-1] >= need:
            take = int(np.searchsorted(cum, need)) + 1
            plan.append((shard["path"], take))
            return plan
        plan.append((shard["path"], len(cum)))
        total += int(cum[-1])
    raise ValueError(f"Train split holds only {total} bytes, budget is {budget_bytes}")


def _iter_docs(
    data_dir: Path, plan: list[tuple[str, int]], batch: int
) -> Iterator[tuple[list[str], list[str], np.ndarray]]:
    for path, take in plan:
        table = pq.read_table(data_dir / path, columns=["id", "text", "n_bytes"]).slice(0, take)
        for start in range(0, take, batch):
            part = table.slice(start, batch)
            yield (
                part.column("id").to_pylist(),
                part.column("text").to_pylist(),
                np.asarray(part.column("n_bytes").to_pylist(), dtype=np.int64),
            )


def cache_key(tokenizer_sha: str, budget_bytes: int, seq_len: int, extra: dict[str, Any]) -> str:
    blob = json.dumps([tokenizer_sha, budget_bytes, seq_len, extra], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def pretokenize_train(
    out_dir: Path,
    tok: Tokenizer,
    ids: SpecialIds,
    data_dir: Path,
    budget_bytes: int,
    seq_len: int,
    batch_docs: int,
) -> dict[str, Any]:
    if ids.vocab_size > np.iinfo(TOKEN_DTYPE).max:
        raise ValueError("Vocab does not fit uint16 storage")
    plan = select_documents(data_dir, budget_bytes)
    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256()
    doc_end_tokens: list[int] = [0]
    doc_end_bytes: list[int] = [0]
    n_tokens = n_bytes = n_docs = 0
    tmp = out_dir / (TOKENS_FILE + ".tmp")
    with tmp.open("wb") as fh:
        for doc_ids, texts, sizes in _iter_docs(data_dir, plan, batch_docs):
            for doc_id in doc_ids:
                fingerprint.update(doc_id.encode() + b"\n")
            encodings = tok.encode_batch_fast(texts, add_special_tokens=False)
            parts = []
            for enc, size in zip(encodings, sizes, strict=True):
                arr = np.asarray(enc.ids, dtype=TOKEN_DTYPE)
                parts.append(arr)
                parts.append(np.asarray([ids.sep], dtype=TOKEN_DTYPE))
                n_tokens += arr.size + 1
                n_bytes += int(size)
                doc_end_tokens.append(n_tokens)
                doc_end_bytes.append(n_bytes)
            np.concatenate(parts).tofile(fh)
            n_docs += len(doc_ids)
    tmp.replace(out_dir / TOKENS_FILE)

    body = seq_len - 2
    n_seqs = n_tokens // body
    boundaries = np.arange(n_seqs + 1, dtype=np.float64) * body
    cum = np.interp(
        boundaries,
        np.asarray(doc_end_tokens, dtype=np.float64),
        np.asarray(doc_end_bytes, dtype=np.float64),
    )
    seq_bytes = np.diff(cum).astype(np.float64)
    np.save(out_dir / SEQ_BYTES_FILE, seq_bytes)
    return {
        "docs": n_docs,
        "doc_ids_sha256": fingerprint.hexdigest(),
        "text_bytes": n_bytes,
        "stream_tokens": n_tokens,
        "bytes_per_token": n_bytes / n_tokens,
        "sequences": int(n_seqs),
        "sequence_bytes": float(seq_bytes.sum()),
        "dropped_tail_tokens": int(n_tokens - n_seqs * body),
    }


def _char_utf8_lengths(text: str) -> np.ndarray:
    cps = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    return 1 + (cps >= 0x80) + (cps >= 0x800) + (cps >= 0x10000)


def token_bytes(text: str, offsets: np.ndarray, doc_bytes: int) -> np.ndarray:
    """Bytes attributed to each token of one document (sums to ``doc_bytes``)."""
    if offsets.size == 0:
        return np.zeros(0, dtype=np.float64)
    cum = np.concatenate([[0], np.cumsum(_char_utf8_lengths(text))])
    span = (cum[offsets[:, 1]] - cum[offsets[:, 0]]).astype(np.float64)
    keys = offsets[:, 0].astype(np.int64) * (len(text) + 1) + offsets[:, 1]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    span /= counts[inverse]
    total = span.sum()
    if total <= 0:
        return np.full(offsets.shape[0], doc_bytes / offsets.shape[0])
    return span * (doc_bytes / total)


def build_eval_set(
    out_dir: Path, tok: Tokenizer, ids: SpecialIds, data_dir: Path, eval_bytes: int, seq_len: int
) -> dict[str, Any]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    table = pq.read_table(data_dir / manifest["heldout"]["path"], columns=["text", "n_bytes"])
    sizes = np.asarray(table.column("n_bytes").to_pylist(), dtype=np.int64)
    take = int(np.searchsorted(np.cumsum(sizes), eval_bytes)) + 1
    texts = table.column("text").to_pylist()[:take]
    encodings = tok.encode_batch(texts, add_special_tokens=False)
    tok_parts: list[np.ndarray] = []
    byte_parts: list[np.ndarray] = []
    for text, enc, size in zip(texts, encodings, sizes[:take], strict=True):
        offsets = np.asarray(enc.offsets, dtype=np.int64).reshape(-1, 2)
        tok_parts += [np.asarray(enc.ids, dtype=TOKEN_DTYPE), np.asarray([ids.sep], TOKEN_DTYPE)]
        byte_parts += [token_bytes(text, offsets, int(size)), np.zeros(1)]
    stream = np.concatenate(tok_parts)
    stream_bytes = np.concatenate(byte_parts)
    body = seq_len - 2
    n_seqs = stream.size // body
    stream[: n_seqs * body].tofile(out_dir / EVAL_TOKENS_FILE)
    stream_bytes[: n_seqs * body].astype(np.float32).tofile(out_dir / EVAL_BYTES_FILE)
    return {
        "docs": take,
        "text_bytes": int(sizes[:take].sum()),
        "sequences": int(n_seqs),
        "tokens": int(n_seqs * body),
    }


def prepare_arm_data(
    pretok_root: Path,
    tokenizer_dir: Path,
    tokenizer_id: str,
    specials: list[str],
    data_dir: Path,
    budget_bytes: int,
    seq_len: int,
    batch_docs: int,
    eval_bytes: int,
) -> tuple[Path, dict[str, Any]]:
    """Build (or reuse) the packed train stream and eval set for one tokenizer."""
    tok, tok_meta = load_tokenizer(tokenizer_dir / tokenizer_id)
    ids = special_ids(tok, specials)
    key = cache_key(
        tok_meta["tokenizer_sha256"],
        budget_bytes,
        seq_len,
        {"eval_bytes": eval_bytes, "heldout": _heldout_sha(data_dir)},
    )
    out_dir = pretok_root / f"{tokenizer_id}-{key}"
    meta_path = out_dir / META_FILE
    if meta_path.exists():
        logger.info("Reusing pre-tokenized data %s", out_dir)
        return out_dir, json.loads(meta_path.read_text())
    logger.info("Pre-tokenizing %s into %s", tokenizer_id, out_dir)
    train = pretokenize_train(out_dir, tok, ids, data_dir, budget_bytes, seq_len, batch_docs)
    evaluation = build_eval_set(out_dir, tok, ids, data_dir, eval_bytes, seq_len)
    meta = {
        "tokenizer_id": tokenizer_id,
        "tokenizer_sha256": tok_meta["tokenizer_sha256"],
        "budget_bytes": budget_bytes,
        "seq_len": seq_len,
        "special_ids": ids.__dict__,
        "train": train,
        "eval": evaluation,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return out_dir, meta


def _heldout_sha(data_dir: Path) -> str:
    return str(json.loads((data_dir / "manifest.json").read_text())["heldout"]["sha256"])


class PackedSequences:
    """Memory-mapped view of a packed stream as (N, seq_len) arrays with [CLS]/[SEP]."""

    def __init__(self, tokens_path: Path, seq_len: int, ids: SpecialIds) -> None:
        self.stream = np.memmap(tokens_path, dtype=TOKEN_DTYPE, mode="r")
        self.body = seq_len - 2
        self.n = self.stream.size // self.body
        self.ids = ids

    def __len__(self) -> int:
        return self.n

    def batch(self, index: np.ndarray) -> np.ndarray:
        starts = index.astype(np.int64)[:, None] * self.body + np.arange(self.body)[None, :]
        out = np.empty((index.size, self.body + 2), dtype=np.int32)
        out[:, 0] = self.ids.cls
        out[:, -1] = self.ids.sep
        out[:, 1:-1] = self.stream[starts]
        return out
