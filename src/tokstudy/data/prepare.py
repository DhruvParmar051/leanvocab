"""Phase 1: stream, NFC-normalize, exact-dedup, split held-out, and shard the corpus.

Usage:
    uv run python -m tokstudy.data.prepare --config configs/data/fineweb_edu_2b.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import stat
import time
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokstudy.data.dedup import ExactDeduper
from tokstudy.data.normalize import nfc
from tokstudy.data.schema import DataPrepConfig, Document
from tokstudy.data.shard import ParquetOptions, ShardWriter, empty_rows, write_rows
from tokstudy.data.split import heldout_key, is_heldout_candidate
from tokstudy.utils.config import config_hash, load_config
from tokstudy.utils.logging import setup_logging
from tokstudy.utils.registry import RunContext, write_metrics
from tokstudy.utils.seeding import seed_everything

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
SHARDS_DIR = "shards"
HELDOUT_DIR = "heldout"
HELDOUT_FILE = "heldout.parquet"
READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH


@dataclass(frozen=True, slots=True)
class _HeldoutCandidate:
    key: int
    doc: Document
    text: str
    n_bytes: int


def _check_clean_output(out_dir: Path) -> None:
    targets = [out_dir / MANIFEST_NAME, out_dir / SHARDS_DIR, out_dir / HELDOUT_DIR]
    occupied = [p for p in targets if p.exists() and (p.is_file() or any(p.iterdir()))]
    if occupied:
        raise FileExistsError(
            f"Output already populated, remove it explicitly first: {[str(p) for p in occupied]}"
        )


def prepare(
    cfg: DataPrepConfig,
    docs: Iterable[Document],
    source_provenance: Callable[[], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the pipeline over ``docs``. Returns (manifest, timing/throughput stats).

    ``source_provenance`` is called after reading and its result is stored in the manifest.
    """
    out_dir = cfg.out_dir
    _check_clean_output(out_dir)
    shards_dir = out_dir / SHARDS_DIR
    heldout_dir = out_dir / HELDOUT_DIR
    shards_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)

    opts = ParquetOptions(
        row_group_rows=cfg.row_group_rows,
        compression=cfg.parquet_compression,
        compression_level=cfg.parquet_compression_level,
    )
    train = ShardWriter(shards_dir, "train", cfg.shard_bytes, opts)
    deduper = ExactDeduper()
    candidates: list[_HeldoutCandidate] = []

    docs_read = bytes_read = tokens_read = nfc_changed = 0
    start = time.monotonic()
    stop_reason = "source_exhausted"
    for doc in docs:
        if cfg.max_docs is not None and docs_read >= cfg.max_docs:
            stop_reason = "max_docs"
            break
        if bytes_read >= cfg.target_text_bytes:
            stop_reason = "target_text_bytes"
            break
        text = nfc(doc.text)
        if text != doc.text:
            nfc_changed += 1
        encoded = text.encode("utf-8")
        n_bytes = len(encoded)
        docs_read += 1
        bytes_read += n_bytes
        tokens_read += doc.token_count_gpt2

        if docs_read % cfg.log_every_docs == 0:
            elapsed = time.monotonic() - start
            logger.info(
                "Read %d docs, %.2f GB (%.1f%% of target), %.1f MB/s",
                docs_read,
                bytes_read / 1e9,
                100 * bytes_read / cfg.target_text_bytes,
                bytes_read / 1e6 / elapsed,
            )

        if deduper.seen(encoded):
            continue
        key = heldout_key(doc.id, cfg.seed)
        if is_heldout_candidate(key, cfg.heldout_fraction):
            candidates.append(_HeldoutCandidate(key, doc, text, n_bytes))
        else:
            train.add(doc.id, text, doc.url, n_bytes, doc.token_count_gpt2)
    train.flush()
    elapsed = time.monotonic() - start

    heldout_entry, heldout_stats = _write_heldout(
        heldout_dir / HELDOUT_FILE, candidates, cfg.heldout_bytes, opts
    )

    shard_entries = train.entries
    manifest: dict[str, Any] = {
        "config_sha256": config_hash(cfg),
        "source": {
            **cfg.source.model_dump(mode="json"),
            **(source_provenance() if source_provenance else {}),
        },
        "normalization": "NFC",
        "dedup": "exact, blake2b-128 over NFC UTF-8 bytes, applied before held-out split",
        "stop_reason": stop_reason,
        "read": {
            "docs": docs_read,
            "text_bytes": bytes_read,
            "token_count_gpt2": tokens_read,
            "docs_changed_by_nfc": nfc_changed,
        },
        "dedup_stats": deduper.stats(),
        "train": {
            "docs": sum(e["docs"] for e in shard_entries),
            "text_bytes": sum(e["text_bytes"] for e in shard_entries),
            "token_count_gpt2": sum(e["token_count_gpt2"] for e in shard_entries),
            "num_shards": len(shard_entries),
            "shards": [{**e, "path": f"{SHARDS_DIR}/{e['path']}"} for e in shard_entries],
        },
        "heldout": {
            **heldout_entry,
            "path": f"{HELDOUT_DIR}/{heldout_entry['path']}",
            **heldout_stats,
            "frozen": True,
        },
    }
    manifest_path = out_dir / MANIFEST_NAME
    with manifest_path.open("x", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    logger.info("Wrote manifest %s", manifest_path)

    timing = {
        "elapsed_sec": elapsed,
        "read_mb_per_sec": bytes_read / 1e6 / elapsed if elapsed else 0.0,
        "docs_per_sec": docs_read / elapsed if elapsed else 0.0,
        "bytes_per_gpt2_token": bytes_read / tokens_read if tokens_read else 0.0,
    }
    return manifest, timing


def _write_heldout(
    path: Path,
    candidates: list[_HeldoutCandidate],
    budget: int,
    opts: ParquetOptions,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep the lowest-hash candidates up to ``budget`` bytes; discard the rest (never train)."""
    candidates.sort(key=lambda c: c.key)
    rows = empty_rows()
    used = 0
    kept = 0
    for cand in candidates:
        if used + cand.n_bytes > budget:
            break
        rows["id"].append(cand.doc.id)
        rows["text"].append(cand.text)
        rows["url"].append(cand.doc.url)
        rows["n_bytes"].append(cand.n_bytes)
        rows["token_count_gpt2"].append(cand.doc.token_count_gpt2)
        used += cand.n_bytes
        kept += 1
    if kept == 0:
        raise RuntimeError("Held-out split is empty; increase heldout_fraction or corpus size")
    discarded = candidates[kept:]
    if not discarded:
        logger.warning(
            "Held-out candidates (%d bytes) did not fill the %d byte budget", used, budget
        )
    entry = write_rows(path, rows, opts)
    path.chmod(READ_ONLY)
    stats = {
        "budget_bytes": budget,
        "candidates": len(candidates),
        "discarded_candidates": len(discarded),
        "discarded_bytes": sum(c.n_bytes for c in discarded),
    }
    return entry, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-docs", type=int, default=None, help="Override max_docs (dry runs)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override out_dir (dry runs)")
    args = parser.parse_args()

    cfg = load_config(args.config, DataPrepConfig)
    overrides: dict[str, Any] = {}
    if args.max_docs is not None:
        overrides["max_docs"] = args.max_docs
    if args.out_dir is not None:
        overrides["out_dir"] = args.out_dir
    if overrides:
        cfg = DataPrepConfig.model_validate({**cfg.model_dump(), **overrides})

    phase = "phase1" if cfg.max_docs is None else "phase1-dryrun"
    ctx = RunContext(phase=phase, config=cfg, seed=cfg.seed)
    setup_logging(ctx.run_dir)
    seed_everything(cfg.seed)
    logger.info("Run %s, config sha256=%s", ctx.run_id, ctx.config_sha256)

    from tokstudy.data.download import HubParquetSource

    source = HubParquetSource(cfg.source, cfg.raw_dir)
    with closing(source.__iter__()) as docs:
        manifest, timing = prepare(cfg, docs, source.provenance)
    timing["fetch_sec"] = {f["path"]: f["fetch_sec"] for f in source.files_used}
    metrics = {
        "manifest_path": str(cfg.out_dir / MANIFEST_NAME),
        "read": manifest["read"],
        "dedup": manifest["dedup_stats"],
        "train": {k: v for k, v in manifest["train"].items() if k != "shards"},
        "heldout": manifest["heldout"],
        "timing": timing,
    }
    write_metrics(ctx, metrics)
    logger.info("Timing: %s", json.dumps(timing))


if __name__ == "__main__":
    main()
