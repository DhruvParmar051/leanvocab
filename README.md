# leanvocab

A controlled study of **small-vocabulary English tokenizers** for sentence-embedding encoders, run end to end on Apple silicon with MLX.

**Claim under test:** a tokenizer with an 8k–16k vocabulary can reach sentence-embedding quality within **≤0.5 MTEB-avg points** of a same-recipe MiniLM-shaped encoder that uses BERT's 30,522-token vocabulary, while freeing **25–38% of the model parameters** and without making end-to-end inference slower.

Everything is held fixed except the tokenizer: the same corpus, the same number of raw text bytes per optimizer step, the same model shape, optimizer, schedule, masking and seed. Cross-tokenizer comparisons use **bits per byte** rather than token-level loss, so arms with different vocabularies stay comparable.

## Status

**This is an ongoing project, currently moving slowly** — the pipeline, the tokenizers and the pretraining runs are in place; the downstream evaluation phases are queued and will land as I find time for them. Nothing below is abandoned.

| Phase | What it does | State |
|---|---|---|
| 1 | Corpus: FineWeb-Edu `sample-10BT`, NFC-normalized, exact-deduped, sharded, frozen held-out split | done |
| 2 | Train 12 tokenizers (BPE / Unigram / WordPiece at 8k/16k/32k, plus BERT vocab and trimmed BERT) | done |
| 3 | Intrinsic evaluation (compression, fertility, Rényi efficiency, truncation, FLOPs, throughput) | done — [`reports/intrinsic.md`](reports/intrinsic.md) |
| 4 | MLM pretraining of 6 arms, 2 GB of text each | 4 of 6 arms done, 2 queued |
| 5 | Contrastive fine-tuning for sentence embeddings | planned, next up |
| 6 | MTEB English subset + latency, final report | planned |

## Results so far

### Phase 3 — intrinsic (held-out split: 10,530 docs, 50.0 MB)

Twelve tokenizers compared on compression, fertility, vocabulary use, truncation and tokenization throughput; the full tables and the Pareto plot are in [`reports/intrinsic.md`](reports/intrinsic.md). Finalists chosen for pretraining: **`bert-orig`** (control), **`bpe-8k`**, **`bpe-16k`**, plus two baselines — factorized embeddings (ALBERT-style, 30,522 × 128) and trimmed BERT vocabularies.

### Phase 4 — MLM pretraining (2 GB of text per arm, one pass, seed 0)

| Arm | Vocab | Params | Held-out bits/byte | vs control | MLM accuracy | Wall clock |
|---|---:|---:|---:|---:|---:|---:|
| `bpe-16k` | 16,000 | 16.89M (−25%) | **1.1613** | −0.032 (better) | 37.6% | 1.79 h |
| `bert-orig` (control) | 30,522 | 22.47M | **1.1931** | — | 35.5% | 2.51 h |
| `bpe-8k` | 8,000 | 13.82M (−38%) | **1.1992** | +0.006 | 38.1% | 1.70 h |
| `factorized-bert-orig` | 30,522 | 14.64M (−35%) | **1.8467** | +0.654 (much worse) | 14.7% | 1.67 h |

Lower bits/byte is better. Each arm saw ~2.00e9 bytes of text in ~6,500 steps of ~307 kB.

Reading of these numbers:

- A 16k BPE vocabulary **beat** the 30,522-vocab control on held-out bits per byte while using 25% fewer parameters and 29% less wall clock.
- An 8k vocabulary landed within 0.006 bits/byte of the control — inside the likely seed-to-seed noise band, which has not yet been measured.
- ALBERT-style embedding factorization **failed** under this shared recipe, so parameter savings from a smaller vocabulary are not interchangeable with savings from factorization.
- Caveats: single seed, and bits per byte is not MTEB. Phases 5–6 are what actually test the claim, and they are still ahead.

Queued next: the two remaining Phase 4 arms (`bert-trim-8k`, `bert-trim-16k`, ~3.5 h), then repeat seeds on the top arms, then Phase 5.

## Hardware and stack

Built for a **MacBook M5 Pro (48 GB), no CUDA**. MLX is the compute path (`mx.fast.scaled_dot_product_attention`, `mx.compile`, `MultiOptimizer` for the decay/no-decay split); PyTorch appears only as a dev dependency for a reference-parity test of the encoder. Peak GPU memory during pretraining was ~24 GB.

Note: MLX enables TF32 matmuls by default on Apple GPUs (~1e-3 relative error, ~1.9× faster). The training configs record `tf32: true` explicitly; the test suite forces it off so numerical-parity tests are meaningful.

## Install

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). All dependency versions are pinned.

```bash
uv sync
```

## Running the phases

Every experiment is driven by a YAML config in `configs/`; there are no magic numbers in the code.

```bash
# Phase 1 — build the corpus (~9.2 GB of text, long-running download)
uv run python -m leanvocab.data.prepare --config configs/data/fineweb_edu_2b.yaml

# Phase 2 — train all tokenizers on the same 5 GB text subset
uv run python -m leanvocab.tokenizers.run_phase2 --config configs/tokenizer/phase2.yaml

# Phase 3 — intrinsic evaluation + report
uv run python -m leanvocab.intrinsic.run_intrinsic --config configs/intrinsic/phase3.yaml

# Phase 4 — pretrain one arm (resumable; finished arms are skipped)
uv run python -m leanvocab.pretrain.run --config configs/pretrain/phase4.yaml --arm bpe-16k

# Phase 4 — all arms in order, finalists first
./scripts/phase4_full_runs.sh

# Phase 4 — 10-minute timed dry run per arm, to estimate full-run cost
./scripts/phase4_dry_runs.sh
```

Tests, lint and types:

```bash
uv run pytest && uv run ruff check . && uv run mypy src tests
```

## Reproducibility

- **Fixed seeds** everywhere; each result records the seed, git SHA (and dirty flag), config SHA-256 and library versions.
- **Results layout:** `results/<phase>/<run_id>/metrics.json` plus a one-line append to `results/index.jsonl`. Writers refuse to overwrite an existing `metrics.json`.
- **Deterministic tokenizers.** The `tokenizers` library is not deterministic out of the box: WordPiece's `limit_alphabet` tie-breaks and `##` insertion follow hash-map order, and Unigram's parallel EM leaves ~1e-4 of float noise. This repo precomputes a shared character alphabet, assigns token ids by a canonical key, and rounds Unigram scores, which makes repeated training runs byte-identical.
- **Deterministic data.** Sharding is byte-budgeted with fixed row groups, atomic renames and a SHA-256 manifest; the held-out split is hash-based, so it does not depend on read order.
- **Dev-split discipline.** MTEB is reserved for final evaluation only. All decisions are made on the frozen held-out split.
- **Tested:** tokenizer round-trips, metric functions against hand-computed values, data-loader determinism, encoder parity against a PyTorch reference, and exact resume-equality for pretraining.

## Repository layout

```
configs/            one YAML per experiment (data, tokenizer, intrinsic, pretrain, eval)
src/leanvocab/
  data/             Phase 1: download, NFC normalize, dedup, split, shard
  tokenizers/       Phase 2: BPE / Unigram / WordPiece training, BERT trimming, round-trips
  intrinsic/        Phase 3: metrics and report generation
  model/            MLX encoder with HF-compatible parameter names
  pretrain/         Phase 4: packing, masking, MLM training loop
  finetune/ eval/   Phases 5-6 (planned)
  utils/            logging, seeding, config loading, run registry, MLX env
scripts/            Phase 4 dry-run and full-run drivers
reports/            generated reports and figures
results/            metrics.json per run + index.jsonl
tests/              pytest suite
```

Large artifacts (`data/`, checkpoints) are not tracked; trained tokenizers live in `artifacts/tokenizers/`.
