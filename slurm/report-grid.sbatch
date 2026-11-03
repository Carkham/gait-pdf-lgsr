#!/usr/bin/env bash
#SBATCH --job-name=gait-grid-report
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/grid-report-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run --locked --group report python scripts/report_grid.py
