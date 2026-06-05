#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=lab_cluster_account
#SBATCH --job-name=nc_sft
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%j_sft.out
#SBATCH --error=logs/%j_sft.err

# 监督微调 SFT
set -e
source /home/dongjunteng/nanochat/.venv/bin/activate

echo "=== Job ${SLURM_JOB_ID}: SFT ==="
echo "Start: $(date)"

torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_sft \
    --depth=20 \
    --run=sft_d20

echo "Done: $(date)"
