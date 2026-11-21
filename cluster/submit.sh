#!/usr/bin/env bash
# Submit a cluster job after showing what it will cost.
#
# Tokens are charged on the requested --time, not the time actually used, and the faculty
# allowance is small (about 1.45 GPU-hours a month). This wrapper prints the balance and
# the estimated cost, and refuses to submit when the balance cannot cover it.
#
#   cluster/submit.sh 00:45:00 "--partition=h200-fullgpu --nodelist=gpunode2 --gres=gpu:h200:1" \
#       --export=ALL,LEANVOCAB_CONFIG=configs/pretrain/phase4_cuda_seed1.yaml cluster/pretrain.sbatch
set -euo pipefail
TOKENS_PER_GPU_HOUR=${TOKENS_PER_GPU_HOUR:-68.8}   # measured 2026-11-21: 275.20 tokens for 4 h
limit=${1:?usage: submit.sh HH:MM:SS "<slurm flags>" <sbatch args...>}
shift
flags=${1:?missing slurm flags (quoted)}
shift

hours=$(awk -F: '{print $1 + $2/60 + $3/3600}' <<<"$limit")
cost=$(awk -v h="$hours" -v r="$TOKENS_PER_GPU_HOUR" 'BEGIN{printf "%.1f", h*r}')
available=$(token-balance 2>/dev/null | awk '/^[a-zA-Z][0-9]+/ {print $(NF-1)}')

echo "requested time : $limit (${hours} h)"
echo "estimated cost : ${cost} tokens"
echo "available      : ${available:-unknown} tokens"
if [ -n "${available:-}" ] && awk -v a="$available" -v c="$cost" 'BEGIN{exit !(a < c)}'; then
  echo "REFUSING: not enough tokens. Reduce --time, or ask the HPC admin for a top-up." >&2
  exit 1
fi
# shellcheck disable=SC2086
exec sbatch --time="$limit" $flags "$@"
