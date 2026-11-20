# Shared cluster settings. Sourced by every cluster script and Slurm job.
#
# Home (/home/$USER) is shared by all nodes but limited to 25 GB; /tmp is node-local, large
# and wiped on reboot. So: code, logs, results and the uv cache live in home; data, the
# virtualenv and checkpoints live in /tmp on the GPU node.
P=leanvocab
export PROJ_REPO=$HOME/gpu-work/$P
export PROJ_STAGE=$HOME/gpu-work/$P-stage
export PROJ_RESULTS=$HOME/gpu-work/$P-results
export PROJ_DATA=/tmp/$USER/$P-data
export PROJ_RUNS=/tmp/$USER/$P-runs

export UV_CACHE_DIR=$HOME/gpu-work/$P-uvcache      # filled on the login node, read on gpunode1
export UV_PROJECT_ENVIRONMENT=/tmp/$USER/$P-venv
export UV_OFFLINE=1                                 # gpunode1 has no usable internet
export UV_PYTHON_DOWNLOADS=never
export HF_HOME=/tmp/$USER/$P-hf-cache
export HF_HUB_OFFLINE=1                             # nothing in Phase 4 needs the network
export HF_HUB_DISABLE_XET=1                         # Xet transfers stall behind the proxy
export PIP_CACHE_DIR=/tmp/$USER/$P-pip-cache
export XDG_CACHE_HOME=/tmp/$USER/$P-cache
export TRITON_CACHE_DIR=/tmp/$USER/$P-cache/triton
export TMPDIR=/tmp/$USER/$P-tmp
export PATH=$HOME/.local/bin:$PATH

# The job writes data and checkpoints under PROJ_DATA/PROJ_RUNS; the config's relative
# paths are redirected there so nothing large lands in home.
export LEANVOCAB_CONFIG=${LEANVOCAB_CONFIG:-configs/pretrain/phase4_cuda.yaml}

mkdir -p "$PROJ_RESULTS" "$PROJ_DATA" "$PROJ_RUNS" "$HF_HOME" "$PIP_CACHE_DIR" \
         "$XDG_CACHE_HOME" "$TMPDIR" "$PROJ_REPO/logs"
chmod 700 "/tmp/$USER" 2>/dev/null || true
