#!/usr/bin/env bash
# Timed Phase 4 dry runs for every arm, one after another (stops at the first failure).
set -euo pipefail
config="${1:-configs/pretrain/phase4.yaml}"
shift || true
arms=("$@")
if [ ${#arms[@]} -eq 0 ]; then
  arms=(bert-orig bpe-8k bpe-16k factorized-bert-orig bert-trim-8k bert-trim-16k)
fi
for arm in "${arms[@]}"; do
  echo "=== dry run: ${arm} ($(date +%H:%M:%S))"
  caffeinate -i uv run python -m leanvocab.pretrain.run --config "$config" --arm "$arm" --dry-run
done
echo "=== all dry runs done ($(date +%H:%M:%S))"
