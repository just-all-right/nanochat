#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=lab_cluster_account
#SBATCH --job-name=nc_eval
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%j_eval.out
#SBATCH --error=logs/%j_eval.err

# 模型评估
set -e
source /home/dongjunteng/nanochat/.venv/bin/activate

echo "=== Job ${SLURM_JOB_ID}: Evaluation ==="
echo "Start: $(date)"

python -m scripts.base_eval --depth=20
python -m scripts.chat_eval --depth=20

echo "Done: $(date)"
