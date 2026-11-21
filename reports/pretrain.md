# Phase 4: MLM pretraining

State as of 2026-11-21. Two sets of runs exist, and they should not be mixed:

- **The CUDA set** (6 arms, seed 0, PyTorch on an H200) is the single-framework result. Use it.
- **The MLX pilot** (4 arms, seed 0, Apple M5 Pro) came first on the hardware the study
  started on. Keep it as a pilot; the two backends do not agree closely enough to pool.

Every arm in a set sees the same documents, the same 2 GB of text in one pass, the same
number of raw text **bytes** per optimizer step, and the same model shape, optimizer,
schedule, masking rate and seed. The tokenizer is the only thing that varies.

## The CUDA set (seed 0, bf16 autocast, H200)

| Arm | Vocab | Encoder params | Held-out bits/byte | vs control | MLM acc | Steps | Wall clock |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bert-trim-16k` | 16,000 | 16.89M (0.75x) | **1.1356** | −0.1052 | 38.6% | 6511 | 3.9 min |
| `bert-trim-8k` | 8,000 | 13.82M (0.62x) | **1.1381** | −0.1026 | 41.6% | 6510 | 4.1 min |
| `bpe-16k` | 16,000 | 16.89M (0.75x) | **1.1823** | −0.0584 | 36.3% | 6502 | 3.9 min |
| `bpe-8k` | 8,000 | 13.82M (0.62x) | **1.2304** | −0.0103 | 37.0% | 6500 | 4.1 min |
| `bert-orig` (control) | 30,522 | 22.47M (1.00x) | **1.2407** | — | 33.2% | 6505 | 4.7 min |
| `factorized-bert-orig` | 30,522 | 14.64M (0.65x) | **1.8328** | +0.5921 | 15.1% | 6505 | 4.4 min |

Lower bits per byte is better. Each arm consumed 2.000e9 bytes of text (+/- 0.02%).

What this says, at this budget and this single seed:

1. **Every small-vocabulary arm beat the 30,522-token control.** The parameter saving did
   not cost quality; it came with a gain.
2. **Trimming BERT's vocabulary beat retraining a tokenizer at the same size.** Baseline B
   (`bert-trim-*`, the most frequent pieces of the original vocabulary on this corpus) was
   the strongest family, ahead of byte-level BPE at both 8k and 16k. That was not the
   expected outcome: Baseline B existed as a "just drop the rare tokens" control.
3. **8k is as good as 16k here.** `bert-trim-8k` is 0.0025 bits/byte behind
   `bert-trim-16k` with 18% fewer parameters, and has the best MLM accuracy of any arm.
4. **Embedding factorization (Baseline A) fails**, by 0.59 bits/byte, on both backends. A
   parameter saving from factorization is not interchangeable with one from a smaller
   vocabulary.

Caveat that governs all of the above: **one seed**. The gap between `bpe-8k` and the
control is 0.010, which no single run can separate from noise. The three-seed spread was
queued and could not run (see "Cluster cost" below), so the error bars are still unknown.

## The MLX pilot, and why the sets are kept apart

| Arm | MLX pilot | CUDA | Difference |
|---|---:|---:|---:|
| `bert-orig` | 1.1931 | 1.2407 | **+0.0476** |
| `bpe-8k` | 1.1992 | 1.2304 | +0.0313 |
| `bpe-16k` | 1.1613 | 1.1823 | +0.0209 |
| `factorized-bert-orig` | 1.8467 | 1.8328 | −0.0139 |

The difference is neither small nor constant, and the ordering changes: the pilot has
`bpe-16k < bert-orig < bpe-8k`, the CUDA set has `bpe-16k < bpe-8k < bert-orig`. Both
backends put those two arms within ~0.01 of each other, so the flip is plausibly noise -
but "plausibly noise" is not a measurement, and until the seed spread exists the two sets
must not be pooled or compared arm-by-arm.

**It is not bf16.** A dry run of `bert-orig` with autocast off reached 1.2414 bits/byte
against 1.2407 with bf16 - a difference of 0.0007, with fp32 taking 0.083 s/step against
0.036. So bf16 is free on this hardware, and the remaining MLX/CUDA gap comes from
somewhere else: different RNG streams (different masked positions and dropout draws) are
the most likely candidate, which makes it the same kind of difference as a different seed.

## How the two backends relate

The data pipeline, the batch plan, the masking rule and the evaluation are shared code. The
encoders are separate implementations held to each other by
`tests/test_torch_encoder_parity.py`: identical parameter trees, and forward outputs within
1e-5 on CPU in float32 for both the plain and factorized shapes. Both write checkpoints
with the same Hugging Face parameter names, and a test loads a torch-written checkpoint
into the MLX model, so published artifacts are backend-independent.

## Cluster cost (read before submitting anything)

The CUDA runs were done on the faculty cluster's H200. Throughput was **1.88M tokens/sec
against 71-79k on the M5 Pro (about 24x)**, so an arm takes ~4 minutes instead of ~2 hours,
and all six took 28 minutes.

The cluster bills **tokens on the `--time` a job requests, not the time it uses**, and the
monthly allowance is 100 tokens, which is about 1.45 GPU-hours (~68.8 tokens per GPU-hour
at 8 CPUs). Submitting a 28-minute run with `--time=08:00:00` cost 550 tokens. The
2026-11 allowance was exhausted this way (743.14 used of 100) and the account was blocked,
which is why the seed sweep did not run. `cluster/submit.sh` now prints the balance and the
estimated cost and refuses to submit what the balance cannot cover; `cluster/README.md`
step 0 documents the rate and sensible limits.

## Artifacts

- Run records: `results/phase4/20261121T*` (CUDA) and `results/phase4/2026091[78]T*` (pilot),
  each with `metrics.json`, `curve.jsonl` and `run.log`.
- CUDA final weights: `data/checkpoints-cuda/<arm>-<key>/model.safetensors` (untracked,
  421 MB), verified by SHA-256 against the cluster before the cluster copies were deleted.
  Optimizer state was left on the cluster and is gone; these runs are complete, so it is
  only needed to extend them.
- The checkpoint directory names carry the training keys as they were computed on
  2026-11-21. `training_key` changed later the same day to include `amp_dtype` and
  `allow_tf32`, so re-running `configs/pretrain/phase4_cuda.yaml` now computes a different
  key and will not find these directories. The recorded `metrics.json` is the link.
- Published on the Hub: the MLX pilot arms only. The CUDA set is not published yet.

## What is not done

- **Seed spread.** `configs/pretrain/phase4_cuda_seed{1,2}.yaml` exist and are ready; they
  need ~70 tokens of cluster allowance (about 1 GPU-hour for 12 arm-runs).
- **The MLX/CUDA gap** is unexplained beyond ruling out bf16.
- **Phases 5 and 6** (contrastive fine-tuning, MTEB, latency) have not started. MTEB
  remains untouched, as the protocol requires.
