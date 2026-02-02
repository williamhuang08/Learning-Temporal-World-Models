#!/bin/bash
#SBATCH --job-name=tawm_sweep
#SBATCH --array 0-15 # create an array of tasks to run the same script
#SBATCH --partition=gpu          
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out  

module load miniconda
module activate myenv
export WANDB_API_KEY="$(cat ../.wandb_key)"

cd /home/wwh2/palmer_scratch/rl  

python - <<'PY'
import numpy as np
import os
import subprocess

task = int(os.environ["SLURM_ARRAY_TASK_ID"])

betas = np.geomspace(1e-3, 1, 4)
gammas = np.geomspace(1e-3, 1, 4)

# Extract (beta, gamma) from task id
i = task // len(gammas)
j = task % len(gammas)

beta = float(betas[i])
gamma = float(gammas[j])

cmd = ["python", "model_oneopt_klbalancing.py", f"--beta {beta}", f"--gamma {gamma}"]
print("Running:", "".join(cmd))
subprocess.check_call(cmd)
PY