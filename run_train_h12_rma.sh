#!/usr/bin/env bash
# Train H1-2 RMA in this repo only. Do not mix PYTHONPATH with homie_h12 — that pulls
# the wrong HIMOnPolicyRunner and fails on RMA_LATENT_DIM / OnPolicyRunner.
#
# Usage (conda env with Isaac Gym + PyTorch, e.g. homieh12rl):
#   cd /path/to/h12_rma
#   conda activate homieh12rl
#   ./run_train_h12_rma.sh [extra train.py args...]
#
# Example:
#   ./run_train_h12_rma.sh --headless --num_envs 512 --rl_device cuda:0 --sim_device cuda:0 --max_iterations 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

# isaacgym python | h12_rma root (legged_gym, rma_modules) | rsl_rl package root — no trailing homie_h12
ISAAC_GYM_PY="${ISAAC_GYM_PY:-$(cd "$SCRIPT_DIR/.." && pwd)/isaacgym/python}"
export PYTHONPATH="${ISAAC_GYM_PY}:${SCRIPT_DIR}:${SCRIPT_DIR}/rsl_rl"

exec python legged_gym/scripts/train.py --task=h1_2_rma --headless"$@"
