"""Deterministic Parquet shard writing."""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("url", pa.string()),
        ("n_bytes", pa.int64()),
        ("token_count_gpt2", pa.int64()),
    ]
)

_HASH_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ParquetOptions:
    row_group_rows: int
    compression: str
    compression_level: int


def write_rows(path: Path, rows: dict[str, list[Any]], opts: ParquetOptions) -> dict[str, Any]:
    """Atomically write ``rows`` to ``path``; return its manifest entry."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    table = pa.Table.from_pydict(rows, schema=SCHEMA)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        table,
        tmp,
        row_group_size=opts.row_group_rows,
        compression=opts.compression,
        compression_level=opts.compression_level,
        use_dictionary=["url"],
        write_statistics=False,
    )
    os.replace(tmp, path)
    entry = {
        "path": path.name,
        "docs": table.num_rows,
        "text_bytes": sum(rows["n_bytes"]),
        "token_count_gpt2": sum(rows["token_count_gpt2"]),
        "file_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    logger.info(
        "Wrote %s: %d docs, %.1f MB text", path.name, entry["docs"], entry["text_bytes"] / 1e6
    )
    return entry


def empty_rows() -> dict[str, list[Any]]:
    return {name: [] for name in SCHEMA.names}


class ShardWriter:
    """Buffers documents and flushes a shard once ``shard_bytes`` of text accumulate."""

    def __init__(self, out_dir: Path, prefix: str, shard_bytes: int, opts: ParquetOptions) -> None:
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_bytes = shard_bytes
        self.opts = opts
        self.entries: list[dict[str, Any]] = []
        self._rows = empty_rows()
        self._buffered_bytes = 0

    def add(self, doc_id: str, text: str, url: str, n_bytes: int, tokens: int) -> None:
        self._rows["id"].append(doc_id)
        self._rows["text"].append(text)
        self._rows["url"].append(url)
        self._rows["n_bytes"].append(n_bytes)
        self._rows["token_count_gpt2"].append(tokens)
        self._buffered_bytes += n_bytes
        if self._buffered_bytes >= self.shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self._rows["id"]:
            return
        path = self.out_dir / f"{self.prefix}-{len(self.entries):05d}.parquet"
        self.entries.append(write_rows(path, self._rows, self.opts))
        self._rows = empty_rows()
        self._buffered_bytes = 0
