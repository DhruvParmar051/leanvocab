# Running leanvocab on the GPU cluster

Phase 4 on the college cluster's RTX PRO 6000 (96 GB), using the PyTorch backend. The MLX
runs in `results/phase4` stay as the pilot; the cluster produces the single-framework set
of all six arms, which is what the final report should use.

Nothing here has been executed yet - these scripts were written against the cluster's
documented layout, not tested on it. Expect to fix small things on the first pass, and run
the smoke test (step 5) before any long job.

## What runs where

| | Login node (`master`) | `gpunode1` |
|---|---|---|
| Internet | yes | no (TLS intercepted) |
| GPU | no | RTX PRO 6000 96 GB + 4x 24 GB MIG |
| Use it for | staging, filling the uv cache, uploading to the Hub | pre-tokenizing, training |

Home is shared and capped at **25 GB**; `/tmp` is node-local, large, and wiped on reboot.
So code, logs, results and the uv cache live in home, while the corpus, the virtualenv,
the pre-tokenized streams and the checkpoints live in `/tmp/$USER/leanvocab-*`.

## 0. Cost: tokens are charged on the time you ASK for

```bash
token-balance
```

The faculty account has a **monthly allowance of 100 tokens**, and a job is billed on its
requested `--time`, not on the time it actually uses. Measured on 2026-11-21: a job with 8
CPUs and 1 GPU quoted **275.20 tokens for a 4-hour limit**, i.e. about **68.8 tokens per
GPU-hour**, so the monthly allowance is roughly **1.45 GPU-hours**. CPU-only jobs are far
cheaper (about 12 tokens per hour at 8 CPUs).

This is easy to get badly wrong. Asking for `--time=08:00:00` for a job that finishes in 28
minutes costs 550 tokens - five months of allowance - and the account is then `BLOCKED`
until an admin tops it up. Request a little more than the work needs and no more:

| Job | Real duration | Sensible `--time` |
|---|---|---|
| `prepare_node` | ~2 min | `00:20:00` |
| One arm (H200) | ~4 min | `00:15:00` |
| All six arms | ~30 min | `01:00:00` |

Use the wrapper, which prints the balance and the estimated cost and refuses to submit when
the balance cannot cover it:

```bash
cluster/submit.sh 01:00:00 "--partition=h200-fullgpu --nodelist=gpunode2 --gres=gpu:h200:1" \
    cluster/pretrain.sbatch
```

## 1. Stage code and corpus (from the Mac)

```bash
./cluster/stage.sh <user>@<login-node>
```

This copies the tracked files, `artifacts/tokenizers/` (~10 MB) and the corpus
(`data/shards/` 3.2 GB + `data/heldout/` 18 MB + `manifest.json`), and records the commit in
`REVISION`. It deliberately does **not** copy `data/raw/` (6.1 GB, only needed to rebuild
shards) or `data/pretok/` (4.4 GB, regenerated on the node). Staged total is about 3.3 GB.

Re-run it after every commit.

## 2. Fill the package cache (login node)

```bash
cd ~/gpu-work/leanvocab && ./cluster/setup_login.sh
```

Downloads Python 3.11 and every locked wheel into `~/gpu-work/leanvocab-uvcache`, which
`gpunode1` then installs from offline. Watch the size it prints: torch plus the CUDA
libraries is several GB against the 25 GB quota.

## 3. Prepare the GPU node

```bash
cluster/submit.sh 00:20:00 "" cluster/prepare_node.sbatch
tail -f logs/leanvocab-prepare-*.out
```

Copies the corpus into `gpunode1:/tmp`, points `~/gpu-work/leanvocab/data` at it with a
symlink (so the configs' relative paths resolve there), installs the environment offline and
prints the torch/CUDA versions.

## 4. Check the CUDA build matches the driver

`uv.lock` resolves torch 2.14 with the **CUDA 13** runtime wheels, which needs a recent
driver. Step 3 prints `cuda build` and `cuda available`; `cuda available False` on a GPU
node usually means the driver is older than the wheels. Fix by pinning a CUDA 12 build in
`pyproject.toml` and re-locking **on the Mac**, then repeating steps 1-3:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
```

## 4b. Pick a node

Both GPU nodes work, and which one is free changes by the hour:

```bash
squeue -w gpunode1 -o "%.8i %.12u %.16P %.10M %.11l %b"   # who holds the card, for how long
sinfo -N -o "%N %P %t %C %G"                               # idle / mix / drain
squeue -u $USER --start                                    # Slurm's estimate for your jobs
```

An interactive job with a multi-day limit can block the full card for days, and Slurm's
estimated start will say so. `gpunode2` carries an H200 (140 GB) and is often idle despite
the cluster notes; it is the better card for this work. Because `/tmp` is node-local, a new
node needs its own prepare job first:

```bash
p=$(sbatch --parsable --nodelist=gpunode2 cluster/prepare_node.sbatch)
sbatch --dependency=afterok:$p -p h200-fullgpu --nodelist=gpunode2 --gres=gpu:h200:1 \
       cluster/pretrain.sbatch
```

Never `scancel` a job that is not yours, however long it has been sitting on the card.

## 5. Smoke test on a MIG slice (do this first)

```bash
sbatch --export=ALL,DRY_RUN=1,ARMS="bpe-16k" \
       -p rtx6000-mig --gres=gpu:1 --time=01:00:00 cluster/pretrain.sbatch
```

A dry run trains for the configured 10 minutes, evaluates, and prints an estimate of the
full-run cost. Note that the CUDA config plans a ~264-sequence batch in a single
micro-batch, which may not fit a 24 GB slice; running the dry run on the whole card is also
the only way to get a timing estimate that means anything for the full runs. Two things to read off it: the estimate (the MLX pilot took 1.7-2.5 h per
arm), and whether `bits_per_byte` is in the same region as the pilot's early evals. A MIG
slice has 24 GB, so a full-batch arm may not fit there - that is expected, and the full-run
job refuses anything under 90 GB on purpose.

## 6. Validate against the pilot before committing to all six arms

The point of the port is that the backends agree. Run one arm for a few hundred steps and
compare its curve with the MLX pilot's at the same step:

```bash
sbatch --export=ALL,ARMS="bpe-16k" cluster/pretrain.sbatch
# then, locally, compare the eval records:
#   jq -c 'select(.type=="eval")' results/phase4/*bpe-16k/../curve.jsonl
```

Identical numbers are not expected: different RNG means different masks, and bf16 differs
from MLX's TF32. Bits per byte within roughly 0.01 at the same step is agreement; a gap of
0.1 or a diverging curve means something in the port is wrong, and the arms should not be
trusted until it is understood.

## 7. Full set

```bash
sbatch cluster/pretrain.sbatch
squeue -u $USER
tail -f logs/leanvocab-phase4-*.out
```

Each arm resumes from its newest checkpoint and is skipped once `final/state.json` exists,
so a requeue or a resubmit continues instead of restarting. Every 15 minutes the job mirrors
`results/` and each arm's final `model.safetensors` into `~/gpu-work/leanvocab-results`,
because `/tmp` can vanish. Optimizer state stays on the node.

## 8. Bring the results back (from the Mac)

```bash
rsync -avP <user>@<login-node>:gpu-work/leanvocab/logs/ ./cluster-logs/
rsync -avP <user>@<login-node>:gpu-work/leanvocab-results/ ./cluster-results/
```

Then publish from the Mac with `leanvocab.publish.hf`, or on the login node (the only node
with internet) using the waiter pattern on the `READY` marker the job writes.

## 9. Clean up when the project is done

```bash
du -sh ~/gpu-work/leanvocab* /tmp/$USER/leanvocab-*          # look first
rm -rf ~/gpu-work/leanvocab-stage                            # staged corpus, once copied
rm -rf /tmp/$USER/leanvocab-venv-fetch                       # login-node leftover
sbatch --partition=cpu --nodelist=gpunode1 --time=00:10:00 \
  --wrap 'rm -rf /tmp/'$USER'/leanvocab-data /tmp/'$USER'/leanvocab-runs /tmp/'$USER'/leanvocab-venv'
```

Only your own files, and only after checking what each directory is.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `backend=torch with device=cuda but CUDA is not available` | ran on a CPU partition, or the driver/wheel mismatch in step 4 |
| `STOP: full runs need the 96 GB card` | got a MIG slice; submit with `--gres=gpu:rtx_pro:1` |
| `uv sync` fails with SSL errors on gpunode1 | cache not filled: re-run step 2 on the login node |
| `STOP: data exists as a real directory` | something wrote `~/gpu-work/leanvocab/data`; move it aside so the symlink can be made |
| Run restarted from step 0 | `/tmp` was wiped; the weights in `leanvocab-results` survive, but the optimizer state does not |
| `Disk quota exceeded` | `du -sh ~/gpu-work/*`; the uv cache and the staged corpus are the big items |
| `has only 0.00 tokens available` / `BLOCKED` | the allowance is spent; see step 0. Only an HPC admin can top it up |
| `job requeued in held state` | the account ran out of tokens while the job was queued |
