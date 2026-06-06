## 实验编号：T0

日期：2026-06-06

项目：nanochat

代码 commit：e1b5982 (fix: 统一 CLI 参数名为连字符格式)

Slurm Job ID：578

GPU 型号：NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB)

GPU 数量：1

模型：depth=6, n_embd=384, n_head=3, n_kv_head=3, vocab_size=32768, window_pattern=L

参数总量：73,531,538（~74M）

数据集：karpathy/climbmix-400b-shuffle（9 个 parquet shards，~900MB）

变量：无（基准跑通实验）

固定参数：

| 参数 | 值 |
|---|---|
| --depth | 6 |
| --max-seq-len | 256 |
| --device-batch-size | 4 |
| --num-iterations | 100 |
| --window-pattern | L |
| --run | dummy |
| --core-metric-every | 0 |
| --sample-every | 0 |

运行命令：

```bash
sbatch slurm_scripts/t0_tiny_pretrain.slurm
```

Slurm 脚本内容：

```bash
#!/bin/bash
#SBATCH --job-name=t0_tiny_pretrain
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%j_t0.out
#SBATCH --error=logs/%j_t0.err

export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
source /home/dongjunteng/nanochat/.venv/bin/activate

# 重新训练 tokenizer
rm -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
python -m scripts.tok_train --max-chars=2000000000

# 训练（后台 GPU 监控）
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
    --format=csv -l 2 > gpu_monitor.log &
python -m scripts.base_train \
    --depth=6 \
    --max-seq-len=256 \
    --device-batch-size=4 \
    --num-iterations=100 \
    --window-pattern=L \
    --core-metric-every=0 \
    --sample-every=0 \
    --run=dummy
```

---

### 指标记录

| 指标 | 结果 |
|---|---|
| 初始 loss | 10.397514 |
| 结束 loss | 5.008587 |
| 平均 step time | 5143.94 ms |
| tokens/s（平均） | 51,580 |
| 显存峰值（torch 分配） | 950.91 MiB |
| 显存峰值（nvidia-smi） | 1,778 MiB |
| GPU 利用率（平均） | 16.9% |
| GPU 利用率（峰值） | 37% |
| 训练总耗时 | 7.58 min |
| 总训练 token 数 | 26,214,400 |
| 最低 validation bpb | 1.523343 |

**补充数据:**

- 自动计算的 global batch size：262,144 tokens
- 每个 micro-batch：1,024 tokens（4 × 256）
- Gradient accumulation steps：256
- 总训练 FLOPs 估计：3.83 × 10^15
- Tokenizer 训练时间：~82 秒（2B chars, 32K vocab）
- 数据下载方式：预先手动下载 9 个 parquet shard 到 `~/.cache/nanochat/base_data_climbmix/`

---

### 观察

1. **Loss 持续下降**：从 10.40 降到 5.01，学习过程正常。100 步内 loss 下降了约 52%。
2. **Step time 波动大**：范围从 ~4,000ms 到 ~7,100ms。第一步因为包含 validation 耗时最长（7,110ms）。后续步中也会因 epoch 切换（pq_idx/rg_idx 变化）出现周期性 spike。
3. **GPU 利用率很低（~17%）**：主要原因有三：
   - 模型太小（depth=6），GPU 计算量远小于 GPU 峰值算力
   - Gradient accumulation = 256，大量 CPU-GPU 同步
   - Flash Attention 3 不可用，回退到 PyTorch SDPA
4. **显存占用极小**（~1GB）：模型只有 74M 参数，远未触及 GPU 96GB 上限。
5. **FA3 不可用**：Blackwell GPU 当前未被 FA3 检测逻辑识别，自动回退到 SDPA。

---

### 解释

- **低 GPU 利用率**的根本原因是"用大炮打蚊子"——在 600W 功耗的 RTX PRO 6000 上跑 74M 参数的小模型。Gradient accumulation=256 意味着每个 optimizer step 需要 256 次 forward/backward，这些都在单卡上串行执行，GPU 大部分时间在等待 CPU 调度和数据搬运。
- **loss 下降速度正常**说明梯度计算、optimizer 更新链路正确。
- **显存占用**：模型权重约 140MB（bf16），加上 optimizer 状态（~280MB for AdamW）、激活值（小 batch 下仅 ~100MB），合计约 520MB + CUDA context 开销，与 950MB 的 torch 分配量吻合。

---

### 代码定位

相关代码文件：

| 链路步骤 | 文件 | 函数/位置 |
|---|---|---|
| 1. 训练入口 | `scripts/base_train.py` | 全局作用域，第 100+ 行起 |
| 2. batch 产生 | `nanochat/dataloader.py` | `tokenizing_distributed_data_loader_with_state_bos_bestfit()` |
| 3. model forward | `nanochat/gpt.py` | `GPT.forward()` (line ~420) |
| 4. loss 计算 | `scripts/base_train.py` | `loss = model(x, y)` (line ~511) |
| 5. backward | `scripts/base_train.py` | `loss.backward()` (line ~517) |
| 6. optimizer.step | `scripts/base_train.py` | `optimizer.step()` (line ~529) |
| 7. tokens/s 统计 | `scripts/base_train.py` | `tok_per_sec` 计算 (line ~552) |
| 8. 日志输出 | `scripts/base_train.py` | `print0(...)` (line ~567) |

---

### 踩坑记录

1. **wandb 认证问题**：Slurm 非交互环境无 TTY，wandb.login 失败。解决：使用 `--run=dummy` 跳过 wandb。
2. **网络不可达**：计算节点无法访问 huggingface.co，数据需提前在登录节点下载到 `~/.cache/nanochat/base_data_climbmix/`。
3. **评估阶段崩溃**：`max-seq-len=256` 时 rotary embedding cache 为 2560，核心评估数据集（hellaswag 等）序列长度超过此值，导致 crash。解决：T0 不跑评估，设置 `--core-metric-every=0 --sample-every=0`。
4. **FA3 检测失败**：Blackwell GPU (SM 120) 的峰值 FLOPS 和 FA3 均未识别，MFU 显示 0%。需要在后续实验中更新 FLOPs 数据库或手动指定。

---

### 下一步

T0 已验证训练链路可跑通。下一步实验 T1：**batch size 扫描**，固定所有其他参数，逐一改变 `--device-batch-size`（2/4/8/16），观察吞吐和显存变化。

---

### 可复现性信息

- Slurm 脚本：`slurm_scripts/t0_tiny_pretrain.slurm`
- 日志文件：`logs/578_t0.out`、`logs/578_t0.err`、`logs/578_gpu_monitor.log`
- 数据位置：`~/.cache/nanochat/base_data_climbmix/`（9 shards）
- Tokenizer：`~/.cache/nanochat/tokenizer/`（vocab 32,768）
- Checkpoint：`~/.cache/nanochat/base_checkpoints/d6/model_000100.pt`
