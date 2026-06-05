#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=lab_cluster_account
#SBATCH --job-name=nc_rl
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%j_rl.out
#SBATCH --error=logs/%j_rl.err

# 强化学习 GRPO
set -e
source /home/dongjunteng/nanochat/.venv/bin/activate

echo "=== Job ${SLURM_JOB_ID}: RL/GRPO ==="
echo "Start: $(date)"

torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_rl \
    --depth=20 \
    --run=rl_d20

echo "Done: $(date)"
