#!/bin/bash
#SBATCH --job-name=t0_tiny_pretrain
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%j_t0.out
#SBATCH --error=logs/%j_t0.err

set -e

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

echo "============================================"
echo "T0: 跑通最小训练链路"
echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"
echo ""

source /home/dongjunteng/nanochat/.venv/bin/activate

# ====== 前置准备：重新训练 tokenizer（数据已下载完毕） ======
echo "=== Step 0: Retraining tokenizer with downloaded data... ==="
rm -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt"
python -m scripts.tok_train --max-chars=2000000000
echo "=== Tokenizer training complete ==="
echo ""

# ====== 训练 ======
# 后台启动 GPU 监控，每 2 秒采样一次
MONITOR_LOG="logs/${SLURM_JOB_ID}_gpu_monitor.log"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv -l 2 > "$MONITOR_LOG" &
MONITOR_PID=$!
echo "GPU monitor PID: ${MONITOR_PID}"

echo ""
echo "=== Training Start ==="

python -m scripts.base_train \
    --depth=6 \
    --max-seq-len=256 \
    --device-batch-size=4 \
    --num-iterations=100 \
    --window-pattern=L \
    --core-metric-every=0 \
    --sample-every=0 \
    --run=dummy

echo ""
echo "=== Training End ==="

# 停止 GPU 监控
kill $MONITOR_PID 2>/dev/null || true

echo ""
echo "============================================"
echo "End: $(date)"
echo "============================================"

# 输出 GPU 监控摘要
echo ""
echo "=== GPU Monitor Summary ==="
cat "$MONITOR_LOG"
