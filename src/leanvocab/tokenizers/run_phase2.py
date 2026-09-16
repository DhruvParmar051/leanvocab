"""Phase 2: train every tokenizer on the same text subset, check round-trips, save.

Usage:
    uv run python -m leanvocab.tokenizers.run_phase2 --config configs/tokenizer/phase2.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from leanvocab.tokenizers.common import (
    SUBSET_FILE,
    Phase2Config,
    RoundtripMode,
    attach_post_processor,
    check_specials,
    compute_alphabet,
    iter_heldout,
    iter_subset_batches,
    save_tokenizer,
    select_subset,
)
from leanvocab.tokenizers.roundtrip import roundtrip_report
from leanvocab.tokenizers.train_bpe import train_bpe
from leanvocab.tokenizers.train_unigram import train_unigram
from leanvocab.tokenizers.train_wordpiece import train_wordpiece
from leanvocab.tokenizers.trim_bert import build_trimmed, count_token_frequencies, load_bert_orig
from leanvocab.utils.config import config_hash, load_config
from leanvocab.utils.logging import setup_logging
from leanvocab.utils.registry import RunContext, write_metrics
from leanvocab.utils.seeding import seed_everything

logger = logging.getLogger(__name__)

BERT_ORIG_ID = "bert-orig"
ALPHABET_FILE = "wordpiece_alphabet.json"


@dataclass(frozen=True)
class TrainSpec:
    tok_id: str
    algorithm: str
    vocab_size: int
    roundtrip: RoundtripMode
    train: Callable[..., Tokenizer]


def _k(v: int) -> str:
    if v % 1000:
        raise ValueError(f"Vocab size {v} is not a multiple of 1000; ids would be ambiguous")
    return f"{v // 1000}k"


def trained_specs(cfg: Phase2Config, alphabet: list[str] | None = None) -> list[TrainSpec]:
    specs: list[TrainSpec] = []
    for v in cfg.bpe.vocab_sizes:
        specs.append(
            TrainSpec(
                f"bpe-{_k(v)}",
                "byte_level_bpe",
                v,
                "exact",
                lambda b, n, v=v: train_bpe(cfg, cfg.bpe, v, b, n),
            )
        )
    for v in cfg.unigram.vocab_sizes:
        specs.append(
            TrainSpec(
                f"uni-{_k(v)}",
                "unigram_byte_fallback",
                v,
                "exact",
                lambda b, n, v=v: train_unigram(cfg, cfg.unigram, v, b, n),
            )
        )
    for v in cfg.wordpiece.vocab_sizes:
        specs.append(
            TrainSpec(
                f"wp-{_k(v)}",
                "wordpiece",
                v,
                "ignore_whitespace",
                lambda b, n, v=v: train_wordpiece(cfg, cfg.wordpiece, v, b, n, _need(alphabet)),
            )
        )
    return specs


def _need(alphabet: list[str] | None) -> list[str]:
    if alphabet is None:
        raise ValueError("WordPiece training requires the precomputed alphabet")
    return alphabet


def _load_or_compute_alphabet(cfg: Phase2Config, subset: dict[str, Any]) -> list[str]:
    path = cfg.out_dir / ALPHABET_FILE
    if path.exists():
        stored = json.loads(path.read_text())
        if stored["subset_doc_ids_sha256"] != subset["doc_ids_sha256"]:
            raise ValueError(f"{path} was computed on a different subset")
        entries = stored["alphabet"]
    else:
        start = time.monotonic()
        entries = compute_alphabet(
            cfg.data_dir, subset, cfg.normalizer, cfg.wordpiece.limit_alphabet
        )
        logger.info("Computed WordPiece alphabet in %.0fs", time.monotonic() - start)
        payload = {"subset_doc_ids_sha256": subset["doc_ids_sha256"], "alphabet": entries}
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    return [e["char"] for e in entries]


def all_ids(cfg: Phase2Config) -> list[str]:
    ids = [s.tok_id for s in trained_specs(cfg)] + [BERT_ORIG_ID]
    ids += [f"bert-trim-{_k(v)}" for v in cfg.bert.trim_sizes]
    return ids


def _write_subset(out_dir: Path, subset: dict[str, Any]) -> None:
    path = out_dir / SUBSET_FILE
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != subset:
            raise ValueError(f"{path} exists and describes a different subset")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(subset, indent=2, sort_keys=True) + "\n")


def _finish(
    cfg: Phase2Config,
    tok: Tokenizer,
    tok_id: str,
    meta: dict[str, Any],
    mode: RoundtripMode,
    heldout: list[str],
    expected_vocab: int,
    check_order: bool = True,
) -> dict[str, Any]:
    if tok.get_vocab_size() != expected_vocab:
        raise ValueError(f"{tok_id}: vocab {tok.get_vocab_size()} != target {expected_vocab}")
    if check_order:
        check_specials(tok, cfg)
    tok.encode_special_tokens = True
    report = roundtrip_report(tok, heldout, mode, cfg.unk_token)
    level = logging.INFO if report["passed"] else logging.ERROR
    logger.log(
        level,
        "%s round-trip (%s): %d/%d matched, %d unk-skipped, %d mismatched",
        tok_id,
        mode,
        report["matched"],
        report["docs"],
        report["unk_docs_skipped"],
        report["mismatched"],
    )
    saved = save_tokenizer(tok, cfg.out_dir / tok_id, {**meta, "id": tok_id, "roundtrip": report})
    return {
        "id": tok_id,
        "vocab_size": saved["vocab_size"],
        "tokenizer_sha256": saved["tokenizer_sha256"],
        "roundtrip_passed": report["passed"],
        "roundtrip_mismatched": report["mismatched"],
        "roundtrip_unk_docs": report["unk_docs_skipped"],
        "sec": meta.get("train_sec"),
    }


def run(cfg: Phase2Config) -> dict[str, Any]:
    wanted = set(cfg.only) if cfg.only else set(all_ids(cfg))
    unknown = wanted - set(all_ids(cfg))
    if unknown:
        raise ValueError(f"Unknown tokenizer ids in `only`: {sorted(unknown)}")
    clashes = [t for t in wanted if (cfg.out_dir / t).exists()]
    if clashes:
        raise FileExistsError(f"Already trained, remove explicitly to retrain: {sorted(clashes)}")

    subset = select_subset(cfg.data_dir, cfg.subset_bytes)
    _write_subset(cfg.out_dir, subset)
    logger.info(
        "Training subset: %d docs, %.2f GB from %d shards",
        subset["docs"],
        subset["text_bytes"] / 1e9,
        len(subset["shards"]),
    )
    heldout = iter_heldout(cfg.data_dir, cfg.roundtrip_docs)
    base_meta = {
        "config_sha256": config_hash(cfg),
        "subset_doc_ids_sha256": subset["doc_ids_sha256"],
        "subset_text_bytes": subset["text_bytes"],
        "special_tokens": cfg.special_tokens,
        "tokenizers_version": metadata.version("tokenizers"),
    }
    results: list[dict[str, Any]] = []

    alphabet = None
    if any(t.startswith("wp-") for t in wanted):
        alphabet = _load_or_compute_alphabet(cfg, subset)

    for spec in trained_specs(cfg, alphabet):
        if spec.tok_id not in wanted:
            continue
        logger.info("Training %s (%s, vocab=%d)", spec.tok_id, spec.algorithm, spec.vocab_size)
        start = time.monotonic()
        tok = spec.train(iter_subset_batches(cfg.data_dir, subset, cfg.batch_docs), subset["docs"])
        train_sec = time.monotonic() - start
        attach_post_processor(tok, cfg)
        logger.info("Trained %s in %.0fs", spec.tok_id, train_sec)
        meta = {
            **base_meta,
            "algorithm": spec.algorithm,
            "vocab_size_target": spec.vocab_size,
            "train_sec": round(train_sec, 1),
        }
        results.append(
            _finish(cfg, tok, spec.tok_id, meta, spec.roundtrip, heldout, spec.vocab_size)
        )

    trim_ids = {f"bert-trim-{_k(v)}": v for v in cfg.bert.trim_sizes}
    if wanted & ({BERT_ORIG_ID} | set(trim_ids)):
        orig, _ = load_bert_orig(cfg)
        bert_meta = {
            **base_meta,
            "source_repo": cfg.bert.repo,
            "source_revision": cfg.bert.revision,
        }
        if BERT_ORIG_ID in wanted:
            results.append(
                _finish(
                    cfg,
                    orig,
                    BERT_ORIG_ID,
                    {**bert_meta, "algorithm": "wordpiece_pretrained", "train_sec": 0.0},
                    "ignore_whitespace",
                    heldout,
                    orig.get_vocab_size(),
                    check_order=False,
                )
            )
        todo = {k: v for k, v in trim_ids.items() if k in wanted}
        if todo:
            logger.info("Counting bert-orig token frequencies on the subset")
            start = time.monotonic()
            orig.encode_special_tokens = True
            counts = count_token_frequencies(
                orig, iter_subset_batches(cfg.data_dir, subset, cfg.batch_docs)
            )
            count_sec = time.monotonic() - start
            logger.info("Counted %d tokens in %.0fs", int(counts.sum()), count_sec)
            np.save(cfg.out_dir / "bert_orig_subset_counts.npy", counts)
            for tok_id, v in todo.items():
                tok, trim_meta = build_trimmed(cfg, orig, counts, v)
                meta = {
                    **bert_meta,
                    **trim_meta,
                    "algorithm": "wordpiece_trimmed",
                    "vocab_size_target": v,
                    "train_sec": round(count_sec, 1),
                }
                results.append(_finish(cfg, tok, tok_id, meta, "ignore_whitespace", heldout, v))

    failed = [r["id"] for r in results if not r["roundtrip_passed"]]
    if failed:
        logger.error("Round-trip FAILED for: %s", failed)
    return {"subset": subset, "tokenizers": results, "roundtrip_failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subset-bytes", type=int, default=None, help="Override (dry runs)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override (dry runs)")
    parser.add_argument("--only", nargs="+", default=None, help="Train only these ids")
    args = parser.parse_args()

    cfg = load_config(args.config, Phase2Config)
    overrides: dict[str, Any] = {}
    if args.subset_bytes is not None:
        overrides["subset_bytes"] = args.subset_bytes
    if args.out_dir is not None:
        overrides["out_dir"] = args.out_dir
    if args.only is not None:
        overrides["only"] = args.only
    if overrides:
        cfg = Phase2Config.model_validate({**cfg.model_dump(), **overrides})

    phase = "phase2" if args.subset_bytes is None and args.out_dir is None else "phase2-dryrun"
    ctx = RunContext(phase=phase, config=cfg, seed=cfg.seed)
    setup_logging(ctx.run_dir)
    seed_everything(cfg.seed)
    logger.info("Run %s, config sha256=%s", ctx.run_id, ctx.config_sha256)
    start = time.monotonic()
    metrics = run(cfg)
    metrics["total_sec"] = round(time.monotonic() - start, 1)
    write_metrics(ctx, metrics)
    if metrics["roundtrip_failed"]:
        raise SystemExit(f"Round-trip failed for {metrics['roundtrip_failed']}")


if __name__ == "__main__":
    main()
