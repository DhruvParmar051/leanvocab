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
| 4 | MLM pretraining of 6 arms, 2 GB of text each | done — all 6 arms on CUDA ([`reports/pretrain.md`](reports/pretrain.md)); 4-arm MLX pilot kept separately |
| 5 | Contrastive fine-tuning for sentence embeddings | planned, next up |
| 6 | MTEB English subset + latency, final report | planned |

## Results so far

### Phase 3 — intrinsic (held-out split: 10,530 docs, 50.0 MB)

Twelve tokenizers compared on compression, fertility, vocabulary use, truncation and tokenization throughput; the full tables and the Pareto plot are in [`reports/intrinsic.md`](reports/intrinsic.md). Finalists chosen for pretraining: **`bert-orig`** (control), **`bpe-8k`**, **`bpe-16k`**, plus two baselines — factorized embeddings (ALBERT-style, 30,522 × 128) and trimmed BERT vocabularies.

### Phase 4 — MLM pretraining (2 GB of text per arm, one pass, seed 0)

All six arms, trained on one H200 with the PyTorch backend — the single-framework set:

| Arm | Vocab | Params | Held-out bits/byte | vs control | MLM acc |
|---|---:|---:|---:|---:|---:|
| `bert-trim-16k` | 16,000 | 16.89M (−25%) | **1.1356** | −0.105 | 38.6% |
| `bert-trim-8k` | 8,000 | 13.82M (−38%) | **1.1381** | −0.103 | 41.6% |
| `bpe-16k` | 16,000 | 16.89M (−25%) | **1.1823** | −0.058 | 36.3% |
| `bpe-8k` | 8,000 | 13.82M (−38%) | **1.2304** | −0.010 | 37.0% |
| `bert-orig` (control) | 30,522 | 22.47M | **1.2407** | — | 33.2% |
| `factorized-bert-orig` | 30,522 | 14.64M (−35%) | **1.8328** | +0.592 | 15.1% |

Lower bits/byte is better; it is loss per byte of raw text, so arms with different
vocabularies are comparable. Each arm took about 4 minutes (1.88M tokens/sec).

Reading of these numbers:

- **Every small-vocabulary arm beat the 30,522-token control.** The parameter saving came
  with a gain rather than a cost.
- **Trimming BERT's vocabulary beat retraining a tokenizer at the same size** — Baseline B
  existed as a "just drop the rare tokens" control and turned out to be the strongest
  family at both sizes.
- **8k is as good as 16k here**, and has the best MLM accuracy of any arm.
- **Embedding factorization (Baseline A) fails** by 0.59 bits/byte, on both backends.
- Caveats: one seed, so the 0.010 gap between `bpe-8k` and the control is not separable
  from noise; the seed spread is configured but unrun. bits/byte is also not MTEB — the
  claim is about sentence embeddings, which is Phases 5–6.

An earlier 4-arm pilot on Apple silicon (MLX) is kept separate: the backends differ by
+0.021 to +0.048 bits/byte and swap the order of two arms, which bf16 was ruled out as the
cause of. See [`reports/pretrain.md`](reports/pretrain.md) for both tables and the detail.

## Released artifacts

The finished checkpoints and every tokenizer are on the Hugging Face Hub. Cards are generated
from the recorded run metrics by [`leanvocab.publish.hf`](src/leanvocab/publish/hf.py), so their
numbers always match `results/`.

| Repo | What it is |
|---|---|
| [`leanvocab-bpe-16k`](https://huggingface.co/DhruvParmar051/leanvocab-bpe-16k) | 16k BPE encoder - best bits/byte so far, 25% fewer parameters than the control |
| [`leanvocab-bpe-8k`](https://huggingface.co/DhruvParmar051/leanvocab-bpe-8k) | 8k BPE encoder - 38% fewer parameters, within 0.006 bits/byte of the control |
| [`leanvocab-bert-orig`](https://huggingface.co/DhruvParmar051/leanvocab-bert-orig) | The control: same recipe with BERT's 30,522-token vocabulary |
| [`leanvocab-factorized-bert-orig`](https://huggingface.co/DhruvParmar051/leanvocab-factorized-bert-orig) | Baseline A, published because it failed; not loadable as a stock BERT |
| [`leanvocab-tokenizers`](https://huggingface.co/DhruvParmar051/leanvocab-tokenizers) | All 12 Phase 2 tokenizers with their intrinsic metrics |

The three non-factorized checkpoints load directly with `transformers`:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("DhruvParmar051/leanvocab-bpe-16k")
tok = AutoTokenizer.from_pretrained("DhruvParmar051/leanvocab-bpe-16k")
```

These are research checkpoints on a 2 GB budget, not finished embedding models - see the
limitations section of any card.

## Hardware and stack

Two interchangeable backends behind one study design:

- **MLX** on a MacBook M5 Pro (48 GB), no CUDA - how Phases 1-4 were run. Peak GPU memory ~24 GB.
- **PyTorch/CUDA** for a GPU cluster - [`cluster/`](cluster/README.md) has the runbook.

The data pipeline, batch plan, masking rule and evaluation are shared; the two encoders are
held to the same outputs by [`tests/test_torch_encoder_parity.py`](tests/test_torch_encoder_parity.py)
(1e-5, both the plain and factorized shapes), and both write checkpoints with the same
Hugging Face parameter names, so artifacts are backend-independent. Pick one with
`backend:` in the config or `--backend` on the command line.

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
  pretrain/         Phase 4: packing, masking, MLM training loop (MLX and PyTorch)
  finetune/ eval/   Phases 5-6 (planned)
  publish/          Hugging Face repo staging and generated model cards
  utils/            logging, seeding, config loading, run registry, MLX env
scripts/            Phase 4 dry-run and full-run drivers
cluster/            GPU-cluster staging, Slurm jobs and runbook
reports/            generated reports and figures
results/            metrics.json per run + index.jsonl
tests/              pytest suite
```

Large artifacts (`data/`, checkpoints) are not tracked; trained tokenizers live in `artifacts/tokenizers/`.
