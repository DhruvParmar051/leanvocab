#!/usr/bin/env bash
# Full Phase 4 runs, finalists first, then baselines. Each arm resumes from its latest
# checkpoint and is skipped once finished, so re-running this script is safe.
set -euo pipefail
config="${1:-configs/pretrain/phase4.yaml}"
shift || true
arms=("$@")
if [ ${#arms[@]} -eq 0 ]; then
  arms=(bert-orig bpe-8k bpe-16k factorized-bert-orig bert-trim-8k bert-trim-16k)
fi
for arm in "${arms[@]}"; do
  if ! pmset -g batt | grep -q "AC Power"; then
    echo "WARNING: on battery power; plug in to avoid sleep and throttling ($(date +%H:%M:%S))"
  fi
  echo "=== full run: ${arm} ($(date +%H:%M:%S))"
  caffeinate -i -s uv run python -m leanvocab.pretrain.run --config "$config" --arm "$arm"
done
echo "=== all full runs done ($(date +%H:%M:%S))"
