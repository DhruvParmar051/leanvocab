#!/usr/bin/env bash
# Run on the LOGIN node (the only node with internet). Fills the home uv cache with the
# Python interpreter and every wheel the GPU node will install offline.
set -euo pipefail
cd "$HOME/gpu-work/leanvocab"
source cluster/env.sh

command -v uv >/dev/null || {
  echo "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh
}

echo "== fetching the interpreter and the locked wheels (needs internet)"
UV_OFFLINE=0 UV_PYTHON_DOWNLOADS=automatic uv python install 3.11
# Cache every group, including dev: the GPU node resolves offline, so anything missing
# here is a job failure there. The dev extras are small next to torch and CUDA.
UV_OFFLINE=0 UV_PYTHON_DOWNLOADS=automatic \
  UV_PROJECT_ENVIRONMENT=/tmp/$USER/leanvocab-venv-fetch \
  uv sync --locked
echo "== uv cache size (counts against the 25 GB home quota)"
du -sh "$UV_CACHE_DIR"
