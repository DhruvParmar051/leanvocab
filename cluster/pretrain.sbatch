#!/bin/bash
# Phase 4 on the full 96 GB card: every arm, resumable, finished arms skipped.
#
#   sbatch cluster/pretrain.sbatch                                  # all arms, gpunode1
#   sbatch --export=ALL,ARMS="bpe-16k bpe-8k" cluster/pretrain.sbatch
#   sbatch --export=ALL,DRY_RUN=1 -p rtx6000-mig --gres=gpu:1 cluster/pretrain.sbatch  # smoke
#
# COST: tokens are charged on the requested --time, not the time used, and the faculty
# allowance is about 1.45 GPU-hours a month (roughly 68.8 tokens per GPU-hour at 8 CPUs).
# Ask for a little more than the work needs and no more: all six arms take ~30 minutes on
# an H200, so --time=01:00:00 is the default here. Check `token-balance` before submitting.
#
# The directives below target gpunode1's 96 GB card; whichever node is free changes by the
# hour, so override them for the H200 (prepare that node first - /tmp is node-local):
#   sbatch --nodelist=gpunode2 cluster/prepare_node.sbatch
#   sbatch -p h200-fullgpu --nodelist=gpunode2 --gres=gpu:h200:1 cluster/pretrain.sbatch
#SBATCH --job-name=leanvocab-phase4
#SBATCH --partition=rtx6000-fullgpu
#SBATCH --gres=gpu:rtx_pro:1        # the FULL 96 GB card; plain gpu:1 can land on a MIG slice
#SBATCH --nodelist=gpunode1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/gpu-work/leanvocab}" || exit 1
source cluster/env.sh
rm -f "$PROJ_RESULTS/READY" "$PROJ_RESULTS/FAILED"
echo "# $(date '+%F %T') | rev $(cat REVISION 2>/dev/null) | $(hostname) | job $SLURM_JOB_ID"
nvidia-smi

dry_run=${DRY_RUN:-0}
arms=${ARMS:-"bert-orig bpe-8k bpe-16k factorized-bert-orig bert-trim-8k bert-trim-16k"}
config=${LEANVOCAB_CONFIG}

# A MIG slice is fine for a smoke test and useless for a full run; fail fast either way.
mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU memory: ${mem} MiB | dry_run=${dry_run} | config=${config}"
if [ "$dry_run" = "0" ] && [ "$mem" -lt 90000 ]; then
  echo "STOP: full runs need a whole card, got ${mem} MiB (a MIG slice?)."
  echo "      Use --gres=gpu:rtx_pro:1 on gpunode1 or --gres=gpu:h200:1 on gpunode2."
  touch "$PROJ_RESULTS/FAILED"; exit 1
fi

# /tmp can be wiped, so mirror the small artifacts (metrics, curves, final weights) into
# home as they appear. Optimizer state is left on the node: it is twice the weight size and
# only needed to resume.
backup() {
  rsync -a --delete "$PROJ_REPO/results/" "$PROJ_RESULTS/results/" 2>/dev/null
  for d in "$PROJ_DATA"/checkpoints/*/final; do
    [ -d "$d" ] || continue
    arm=$(basename "$(dirname "$d")")
    mkdir -p "$PROJ_RESULTS/checkpoints/$arm"
    rsync -a "$d/model.safetensors" "$d/state.json" "$PROJ_RESULTS/checkpoints/$arm/" 2>/dev/null
  done
}
( while sleep 900; do backup; done ) &
backup_pid=$!

status=0
for arm in $arms; do
  echo "=== $arm ($(date '+%H:%M:%S'))"
  flags=""
  [ "$dry_run" = "1" ] && flags="--dry-run"
  # shellcheck disable=SC2086
  /usr/bin/time -p uv run python -m leanvocab.pretrain.run \
      --config "$config" --arm "$arm" $flags
  rc=$?
  echo "=== $arm exit $rc ($(date '+%H:%M:%S'))"
  if [ $rc -ne 0 ]; then status=$rc; break; fi
done

kill $backup_pid 2>/dev/null
backup
if [ $status -eq 0 ]; then
  touch "$PROJ_RESULTS/READY"
else
  touch "$PROJ_RESULTS/FAILED"
fi
echo "all arms done, exit $status ($(date '+%F %T'))"
exit $status
