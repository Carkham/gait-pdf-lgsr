#!/usr/bin/env bash
#SBATCH --job-name=gait-ablations
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/ablations-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
unset LD_LIBRARY_PATH CUDA_HOME
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run --locked python -u -m gait.run_ablations --variant-index "${SLURM_ARRAY_TASK_ID:-0}" "$@"
