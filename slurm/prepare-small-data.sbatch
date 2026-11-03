#!/usr/bin/env bash
#SBATCH --job-name=gait-small-data
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/small-data-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset LD_LIBRARY_PATH CUDA_HOME
export PATH="$HOME/.local/bin:$PATH"
uv run --locked python scripts/prepare_small_data.py
