"""Phase 3: intrinsic evaluation of every Phase 2 tokenizer on the held-out split.

Usage:
    uv run python -m leanvocab.intrinsic.run_intrinsic --config configs/intrinsic/phase3.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from pydantic import Field
from tokenizers import Tokenizer

from leanvocab.intrinsic.metrics import (
    EncoderShape,
    renyi_efficiency,
    split_paragraphs,
    tokens_per_word,
    truncated_share,
    word_spans,
)
from leanvocab.intrinsic.report import write_report
from leanvocab.tokenizers.common import load_tokenizer
from leanvocab.utils.config import StrictConfig, load_config
from leanvocab.utils.logging import setup_logging
from leanvocab.utils.registry import RunContext, write_metrics
from leanvocab.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


class ThroughputConfig(StrictConfig):
    single_thread_docs: int = Field(gt=0)
    repeats: int = Field(ge=1)
    batch_docs: int = Field(gt=0)


class EncoderConfig(StrictConfig):
    n_layers: int = Field(gt=0)
    d_model: int = Field(gt=0)
    d_ffn: int = Field(gt=0)
    max_len: int = Field(gt=0)
    type_vocab: int = Field(gt=0)


class GateConfig(StrictConfig):
    max_len_ratio: float = Field(gt=1.0)
    candidate_prefixes: list[str]
    claim_vocab_sizes: list[int]


class Phase3Config(StrictConfig):
    seed: int = Field(ge=0)
    data_dir: Path
    tokenizer_dir: Path
    report_dir: Path
    tokenizers: list[str]
    reference: str
    renyi_alpha: float = Field(gt=0.0)
    max_lens: list[int]
    n_special_tokens: int = Field(ge=0)
    min_paragraph_chars: int = Field(ge=1)
    encode_batch_docs: int = Field(gt=0)
    split_word_min_pieces: int = Field(ge=2)
    throughput: ThroughputConfig
    encoder: EncoderConfig
    gate: GateConfig


def _load_heldout(data_dir: Path) -> tuple[list[str], np.ndarray, str]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    heldout = manifest["heldout"]
    table = pq.read_table(data_dir / heldout["path"], columns=["text", "n_bytes"])
    texts: list[str] = table.column("text").to_pylist()
    n_bytes = np.asarray(table.column("n_bytes").to_pylist(), dtype=np.int64)
    return texts, n_bytes, str(heldout["sha256"])


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _throughput(tok: Tokenizer, docs: list[str], cfg: Phase3Config) -> dict[str, float]:
    single = docs[: cfg.throughput.single_thread_docs]
    single_mb = sum(len(d.encode("utf-8")) for d in single) / 1e6
    all_mb = sum(len(d.encode("utf-8")) for d in docs) / 1e6

    def run_single() -> float:
        start = time.perf_counter()
        for d in single:
            tok.encode(d, add_special_tokens=False)
        return time.perf_counter() - start

    def run_batched() -> float:
        start = time.perf_counter()
        for chunk in _chunks(docs, cfg.throughput.batch_docs):
            tok.encode_batch(chunk, add_special_tokens=False)
        return time.perf_counter() - start

    run_single()
    single_sec = statistics.median(run_single() for _ in range(cfg.throughput.repeats))
    run_batched()
    batched_sec = statistics.median(run_batched() for _ in range(cfg.throughput.repeats))
    return {
        "single_thread_docs_per_sec": len(single) / single_sec,
        "single_thread_mb_per_sec": single_mb / single_sec,
        "batched_docs_per_sec": len(docs) / batched_sec,
        "batched_mb_per_sec": all_mb / batched_sec,
    }


def evaluate(
    tok: Tokenizer,
    docs: list[str],
    doc_bytes: np.ndarray,
    spans: list[tuple[np.ndarray, np.ndarray]],
    paragraphs: list[str],
    cfg: Phase3Config,
    shape: EncoderShape,
    unk_token: str,
) -> dict[str, Any]:
    vocab_size = tok.get_vocab_size()
    unk_id = tok.token_to_id(unk_token)
    counts = np.zeros(vocab_size, dtype=np.int64)
    lengths = np.zeros(len(docs), dtype=np.int64)
    words = words_split = words_unk = whitespace_tokens = unk_tokens = docs_unk = 0

    doc_index = 0
    for chunk in _chunks(docs, cfg.encode_batch_docs):
        encodings = tok.encode_batch(chunk, add_special_tokens=False)
        chunk_ids = []
        for enc in encodings:
            ids = np.asarray(enc.ids, dtype=np.int64)
            offsets = np.asarray(enc.offsets, dtype=np.int64).reshape(-1, 2)
            starts, ends = spans[doc_index]
            assigned, per_word = tokens_per_word(offsets, starts, ends)
            lengths[doc_index] = ids.size
            words += starts.size
            words_split += int((per_word >= cfg.split_word_min_pieces).sum())
            whitespace_tokens += int((assigned < 0).sum())
            if unk_id is not None:
                is_unk = ids == unk_id
                n_unk = int(is_unk.sum())
                if n_unk:
                    unk_tokens += n_unk
                    docs_unk += 1
                    words_unk += int(np.unique(assigned[is_unk & (assigned >= 0)]).size)
            chunk_ids.append(ids)
            doc_index += 1
        counts += np.bincount(np.concatenate(chunk_ids), minlength=vocab_size)

    total_tokens = int(lengths.sum())
    para_lengths = np.asarray(
        [
            len(e.ids)
            for chunk in _chunks(paragraphs, cfg.encode_batch_docs)
            for e in tok.encode_batch_fast(chunk, add_special_tokens=False)
        ],
        dtype=np.int64,
    )
    capped = np.minimum(lengths + cfg.n_special_tokens, shape.max_len)
    para_capped = np.minimum(para_lengths + cfg.n_special_tokens, shape.max_len)
    observed = int((counts > 0).sum())
    return {
        "vocab_size": vocab_size,
        "token_embedding_params": shape.token_embedding_params(vocab_size),
        "total_params": shape.total_params(vocab_size),
        "tokens": total_tokens,
        "bytes_per_token": float(doc_bytes.sum() / total_tokens),
        "fertility": total_tokens / words,
        "words": words,
        "share_words_split_ge_min_pieces": words_split / words,
        "whitespace_only_tokens": whitespace_tokens,
        "renyi_efficiency": renyi_efficiency(counts, cfg.renyi_alpha, vocab_size),
        "renyi_efficiency_observed_types": renyi_efficiency(counts, cfg.renyi_alpha, observed),
        "observed_types": observed,
        "vocab_used_share": observed / vocab_size,
        "unk_token_rate": unk_tokens / total_tokens,
        "unk_word_rate": words_unk / words,
        "unk_doc_share": docs_unk / len(docs),
        "doc_len_mean": float(lengths.mean()),
        "doc_len_p50": float(np.median(lengths)),
        "doc_len_p95": float(np.percentile(lengths, 95)),
        "doc_truncated": {
            str(m): truncated_share(lengths, m, cfg.n_special_tokens) for m in cfg.max_lens
        },
        "para_len_mean": float(para_lengths.mean()),
        "para_len_p95": float(np.percentile(para_lengths, 95)),
        "para_truncated": {
            str(m): truncated_share(para_lengths, m, cfg.n_special_tokens) for m in cfg.max_lens
        },
        "flops_per_doc_at_max_len": float(shape.forward_flops(capped).mean()),
        "flops_per_para": float(shape.forward_flops(para_capped).mean()),
        "throughput": _throughput(tok, docs, cfg),
    }


def apply_gate(results: dict[str, dict[str, Any]], cfg: Phase3Config) -> dict[str, Any]:
    ref = results[cfg.reference]
    for r in results.values():
        r["len_ratio_vs_reference"] = r["doc_len_mean"] / ref["doc_len_mean"]
        r["flops_doc_ratio_vs_reference"] = (
            r["flops_per_doc_at_max_len"] / ref["flops_per_doc_at_max_len"]
        )
        r["flops_para_ratio_vs_reference"] = r["flops_per_para"] / ref["flops_per_para"]
        r["params_ratio_vs_reference"] = r["total_params"] / ref["total_params"]
        r["passes_len_gate"] = r["len_ratio_vs_reference"] <= cfg.gate.max_len_ratio
    proposal: dict[str, Any] = {"reference": cfg.reference, "by_vocab_size": {}}
    for v in cfg.gate.claim_vocab_sizes:
        pool = [
            (tid, r)
            for tid, r in results.items()
            if r["vocab_size"] == v and tid.startswith(tuple(cfg.gate.candidate_prefixes))
        ]
        passing = sorted(
            (p for p in pool if p[1]["passes_len_gate"]),
            key=lambda p: (-p[1]["bytes_per_token"], -p[1]["renyi_efficiency"]),
        )
        proposal["by_vocab_size"][str(v)] = {
            "ranked_passing": [tid for tid, _ in passing],
            "dropped_by_gate": sorted(tid for tid, r in pool if not r["passes_len_gate"]),
        }
    return proposal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, Phase3Config)
    ctx = RunContext(phase="phase3", config=cfg, seed=cfg.seed)
    setup_logging(ctx.run_dir)
    seed_everything(cfg.seed)
    logger.info("Run %s, config sha256=%s", ctx.run_id, ctx.config_sha256)
    if cfg.reference not in cfg.tokenizers:
        raise ValueError(f"Reference {cfg.reference} is not in the tokenizer list")

    shape = EncoderShape(**cfg.encoder.model_dump())
    docs, doc_bytes, heldout_sha = _load_heldout(cfg.data_dir)
    spans = [word_spans(d) for d in docs]
    paragraphs = [p for d in docs for p in split_paragraphs(d, cfg.min_paragraph_chars)]
    logger.info(
        "Held-out: %d docs, %.1f MB, %d words, %d paragraphs",
        len(docs),
        doc_bytes.sum() / 1e6,
        sum(s.size for s, _ in spans),
        len(paragraphs),
    )

    results: dict[str, dict[str, Any]] = {}
    for tid in cfg.tokenizers:
        start = time.monotonic()
        tok, meta = load_tokenizer(cfg.tokenizer_dir / tid)
        unk = meta["special_tokens"][1]
        results[tid] = {
            "tokenizer_sha256": meta["tokenizer_sha256"],
            **evaluate(tok, docs, doc_bytes, spans, paragraphs, cfg, shape, unk),
        }
        r = results[tid]
        logger.info(
            "%s: %.3f bytes/tok, fertility %.3f, renyi %.4f, unk %.2e, mean len %.0f (%.0fs)",
            tid,
            r["bytes_per_token"],
            r["fertility"],
            r["renyi_efficiency"],
            r["unk_token_rate"],
            r["doc_len_mean"],
            time.monotonic() - start,
        )

    proposal = apply_gate(results, cfg)
    env = {
        "rayon_threads": os.environ.get("RAYON_RS_NUM_CPUS", "default"),
        "cpu_count": os.cpu_count(),
    }
    heldout = {
        "sha256": heldout_sha,
        "docs": len(docs),
        "bytes": int(doc_bytes.sum()),
        "words": int(sum(s.size for s, _ in spans)),
        "paragraphs": len(paragraphs),
    }
    report = write_report(cfg.report_dir, results, proposal, cfg, shape, heldout, ctx.run_id)
    write_metrics(
        ctx,
        {
            "heldout": heldout,
            "environment": env,
            "tokenizers": results,
            "proposal": proposal,
            "report": str(report),
        },
    )


if __name__ == "__main__":
    main()
