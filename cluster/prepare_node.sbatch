#!/bin/bash
# One CPU job on gpunode1: copy the corpus into that node's /tmp and build the offline
# virtualenv there. Run cluster/setup_login.sh on the login node first (it fills the cache).
#
#   sbatch cluster/prepare_node.sbatch
#SBATCH --job-name=leanvocab-prepare
#SBATCH --partition=cpu
#SBATCH --nodelist=gpunode1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/gpu-work/leanvocab}" || exit 1
source cluster/env.sh
echo "# $(date '+%F %T') | rev $(cat REVISION 2>/dev/null) | $(hostname) | job $SLURM_JOB_ID"

# The config's paths are relative, so point the repo's data/ at this node's /tmp. Everything
# large the run produces (pre-tokenized streams, checkpoints) then lands outside the quota.
if [ ! -L data ] && [ -d data ]; then
  echo "STOP: $PROJ_REPO/data exists as a real directory; move it aside first"; exit 1
fi
ln -sfn "$PROJ_DATA" data
echo "data -> $(readlink data)"

echo "== corpus into node-local /tmp"
rsync -a --info=stats2 "$PROJ_STAGE/data/" "$PROJ_DATA/" || exit 1
du -sh "$PROJ_DATA"/*

echo "== offline environment"
# --no-default-groups, not --no-dev: the latter still put the dev group in the install plan.
uv sync --locked --no-default-groups || exit 1
uv run python - <<'PY' || exit 1
import platform

import torch

import leanvocab
print("python", platform.python_version(), "| torch", torch.__version__)
print("cuda build", torch.version.cuda, "| cuda available", torch.cuda.is_available())
print("leanvocab", leanvocab.__file__)
PY
echo "prepared OK"
