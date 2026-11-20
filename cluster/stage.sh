#!/usr/bin/env bash
# Run on the Mac. Copies the tracked code, the tokenizers and the corpus shards to the
# cluster's home, and records the commit the cluster is running.
#
#   ./cluster/stage.sh <user>@<login-node>
#
# Staged size: shards ~3.2 GB + held-out 18 MB + tokenizers ~10 MB, against a 25 GB quota.
# data/raw (6.1 GB) is deliberately not staged: it is only needed to rebuild the shards,
# and data/pretok is regenerated on the GPU node instead of being copied.
set -euo pipefail
cluster=${1:?usage: cluster/stage.sh <user>@<login-node>}
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"

if [ -n "$(git status --porcelain)" ]; then
  echo "WARNING: working tree is dirty; the cluster will run uncommitted code" >&2
fi

ssh -o BatchMode=yes "$cluster" "mkdir -p gpu-work/leanvocab/logs gpu-work/leanvocab-stage"
echo "== code (tracked files only)"
git ls-files -z | rsync -av --from0 --files-from=- ./ "$cluster:gpu-work/leanvocab/"
git rev-parse HEAD | ssh -o BatchMode=yes "$cluster" "cat > gpu-work/leanvocab/REVISION"

echo "== tokenizers (small, straight into the repo in home)"
rsync -avP artifacts/ "$cluster:gpu-work/leanvocab/artifacts/"

echo "== corpus (staged in home, copied to the GPU node's /tmp by prepare_node.sbatch)"
rsync -avP data/manifest.json "$cluster:gpu-work/leanvocab-stage/data/"
rsync -avP data/heldout/ "$cluster:gpu-work/leanvocab-stage/data/heldout/"
rsync -avP data/shards/ "$cluster:gpu-work/leanvocab-stage/data/shards/"

echo "== quota after staging"
ssh -o BatchMode=yes "$cluster" "quota -s 2>/dev/null || df -h ~"
