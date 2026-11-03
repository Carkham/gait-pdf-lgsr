#!/usr/bin/env bash
#SBATCH --job-name=gait-uv-setup
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/setup-%j.log

set -euo pipefail
unset LD_LIBRARY_PATH CUDA_HOME
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked
uv run --locked python - <<'PY'
import os
import torch
import swanlab

gpu = torch.cuda.get_device_name(0)
print("PyTorch", torch.__version__, "CUDA", torch.version.cuda)
print("GPU", gpu, "SwanLab", swanlab.__version__)
run = swanlab.init(
    project="gait-pdf-lgsr",
    experiment_name="environment-" + os.environ["SLURM_JOB_ID"],
    job_type="environment",
    description="Environment provenance only; no gait recognition experiment or accuracy.",
    config={
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "swanlab": swanlab.__version__,
        "gpu": gpu,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "partition": os.environ["SLURM_JOB_PARTITION"],
    },
)
run.log({"environment/gpu_available": 1})
run.finish()
PY
