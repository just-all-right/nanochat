# NanoChat 全栈大语言模型训练与部署 —— 实习项目实验指导书

---

> **适用岗位方向**：大模型训练工程师、LLM 推理优化工程师、NLP 算法工程师、AI Infra 工程师
>
> **实习周期**：8 周（可根据实际情况调整至 6-12 周）
>
> **前置技能要求**：Python 熟练、PyTorch 基础、Transformer 原理了解、Linux 命令行基本使用
>
> **日期**：2026 年 6 月

---

## 目录

1. [项目概述与学习目标](#1-项目概述与学习目标)
2. [项目背景与行业价值](#2-项目背景与行业价值)
3. [实验环境准备](#3-实验环境准备)
4. [实验一：分词器训练与评估](#4-实验一分词器训练与评估)
5. [实验二：GPT 模型架构深度解析](#5-实验二gpt-模型架构深度解析)
6. [实验三：数据管线与高效数据加载](#6-实验三数据管线与高效数据加载)
7. [实验四：基础模型预训练](#7-实验四基础模型预训练)
8. [实验五：混合优化器与分布式训练](#8-实验五混合优化器与分布式训练)
9. [实验六：监督微调 SFT](#9-实验六监督微调-sft)
10. [实验七：强化学习 RLHF/GRPO](#10-实验七强化学习-rlhfgpro)
11. [实验八：推理引擎与模型服务化](#11-实验八推理引擎与模型服务化)
12. [实验九：模型评估体系](#12-实验九模型评估体系)
13. [实验十：全流程整合与报告生成](#13-实验十全流程整合与报告生成)
14. [能力图谱与求职映射](#14-能力图谱与求职映射)
15. [评估标准](#15-评估标准)
16. [附录：常见问题与进阶方向](#16-附录常见问题与进阶方向)

---

## 1. 项目概述与学习目标

### 1.1 项目简介

NanoChat 是由前特斯拉 AI 总监、OpenAI 创始成员 Andrej Karpathy 设计的最小化全栈大语言模型（LLM）训练框架。该项目以极简的代码量（约 10,000 行 Python）覆盖了从零训练一个 GPT-2 级别对话模型的完整流程，包括：

- **分词器训练**（BPE + tiktoken 推理优化）
- **预训练**（支持缩放定律自动推导超参数）
- **监督微调**（SFT，多任务混合数据）
- **强化学习**（GRPO/REINFORCE，基于数学推理）
- **推理优化**（KV Cache、模型量化 FP8、Flash Attention 3）
- **模型服务**（FastAPI 流式部署、多 GPU 工作池）
- **评估体系**（CORE 基准、BPB 评估、ChatCORE 综合评分）

**关键数据**：在 8×H100 GPU 上，约 2 小时即可训练出超越 GPT-2（2019 年训练成本约 43,000 美元）水平的模型，按需实例成本约 48 美元，竞价实例仅约 15 美元。

### 1.2 学习目标

完成本实习项目后，你将具备以下能力：

| 能力维度 | 具体技能 |
|---------|---------|
| **模型架构** | 深入理解 GPT/Transformer 的 14 项现代改进（RoPE、GQA、QK Norm、ReLU²、值残差、滑动窗口注意力等） |
| **训练工程** | 掌握预训练→SFT→RL 三阶段训练流程，理解缩放定律自动配置超参数的方法 |
| **分布式训练** | 实践 DDP + ZeRO-2 风格优化器分片，理解多卡通信模式 |
| **训练加速** | 掌握 FP8 混合精度训练、torch.compile 加速、Flash Attention 3 的使用 |
| **优化器设计** | 深入理解 Muon（动量正交化）+ AdamW 混合优化器原理与实现 |
| **数据工程** | 掌握 BOS 对齐 Best-Fit 打包、Token 级掩码、多任务混合数据策略 |
| **推理优化** | 掌握 KV Cache 管理、流式推理、推理引擎设计 |
| **模型评估** | 掌握 BPB、CORE 基准、ChatCORE、Pass@k 等多维度评估方法 |
| **模型部署** | 掌握 FastAPI + SSE 流式服务、多 GPU 负载均衡、Web UI 搭建 |

### 1.3 项目在求职中的亮点

- ✅ **全栈能力证明**：覆盖"数据→训练→评估→部署"全链路，展示你不仅是"调参侠"
- ✅ **代码质量标杆**：学习 Karpathy 级别的工程代码风格，面试中可展示代码品味
- ✅ **前沿技术实践**：Flash Attention 3、FP8、Muon 优化器、GRPO 均为 2024-2025 年业界前沿
- ✅ **成本意识**：理解计算最优训练（Chinchilla 缩放定律），具备资源规划能力
- ✅ **可复现结果**：训练出可对话的 GPT-2 级别模型，面试 Demo 有说服力

---

## 2. 项目背景与行业价值

### 2.1 行业背景

截至 2026 年，大语言模型已成为 AI 产业的核心基础设施。从 ChatGPT 到 Claude，从代码助手到 AI 搜索，LLM 正在重塑软件行业。然而，真正具备"从零训练+部署"全栈能力的工程师仍然稀缺。大多数从业者停留在"调用 API"或"微调开源模型"层面。

本项目的设计哲学是：**用一个复杂度旋钮（`--depth`）控制模型规模，所有其他超参数通过 Chinchilla 缩放定律自动推导**。这体现了业界顶级工程师的系统设计思维——复杂系统的简洁接口。

### 2.2 技术栈与行业对标

| 组件 | NanoChat 实现 | 行业对标 |
|------|-------------|---------|
| 预训练 | `torch.compile` + FP8 + DDP | GPT-4/Claude 训练管线简化版 |
| 注意力机制 | Flash Attention 3 (Hopper) | vLLM、TensorRT-LLM 推理引擎 |
| 优化器 | Muon (Polar Express) + AdamW | DeepSpeed ZeRO-2、Megatron-LM |
| 强化学习 | GRPO (无 KL 惩罚简化版) | DeepSeek-R1 同款算法 |
| 推理服务 | FastAPI + SSE 流式 | vLLM OpenAI-compatible Server |
| 评估 | CORE (DCLM 论文) + BPB + ChatCORE | HELM、OpenLLMLeaderboard |

---

## 3. 实验环境准备

> 📖 **完整的环境配置指南请阅读**：[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)
>
> 本章仅提供概览。详细的 Slurm 脚本模板、排错指南、命令速查表等请参见上述文档。

### 3.1 集群硬件

| GPU | 数量 | 单卡显存 | CPU | 内存 | 个人配额 |
|-----|------|---------|-----|------|---------|
| NVIDIA RTX PRO 6000 Blackwell | 8 | ~96 GB | 112 核 | ~386 GB | ⚠️ **最多 2 张** |

### 3.2 核心规则

```
╔════════════════════════════════════════════════════════════╗
║  ⚠️  GPU 必须通过 Slurm 调度使用，禁止裸机直接运行！    ║
║                                                          ║
║  登录节点 → 写代码、Git、看文件、提交任务、CPU 测试      ║
║  计算节点 → 所有 GPU 程序（训练、推理、评估）            ║
╚════════════════════════════════════════════════════════════╝
```

### 3.3 环境初始化速查

```bash
# 1. 进入项目并激活虚拟环境
cd /home/dongjunteng/nanochat
source .venv/bin/activate

# 2. 验证环境（CPU 部分）
python -c "from nanochat.gpt import GPT, GPTConfig; print('nanochat OK')"

# 3. 验证 Slurm + GPU（会进入计算节点执行）
srun --gres=gpu:1 --time=00:05:00 bash -c '
source /home/dongjunteng/nanochat/.venv/bin/activate
python -c "import torch; print(f\"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}\")"
'

# 4. 创建实验日志目录
mkdir -p logs slurm_scripts
```

### 3.4 两种使用方式

| 方式 | 命令 | 适用场景 |
|------|------|---------|
| **交互式** | `srun --gres=gpu:1 --time=02:00:00 --pty bash` | 调试代码、短实验、探索分析 |
| **批量提交** | `sbatch slurm_scripts/xxx.slurm` | 长时间训练、无人值守运行 |

交互式示例：

```bash
# 申请 1 GPU 的交互 shell
srun --partition=compute --account=lab_cluster_account \
     --gres=gpu:1 --time=02:00:00 \
     --ntasks=1 --cpus-per-task=8 --mem=64G --pty bash

# 进入后：
source /home/dongjunteng/nanochat/.venv/bin/activate
python -m scripts.base_train --depth=6 --max-seq-len=256 \
    --device-batch-size=4 --num-iterations=200
# 退出: Ctrl+D 或 exit（资源立即释放）
```

批量提交示例：

```bash
# 编写脚本 slurm_scripts/train.slurm（模板见 ENVIRONMENT_SETUP.md 第 11 章）
sbatch slurm_scripts/train.slurm

# 监控
squeue -u $(whoami)
tail -f logs/<JOB_ID>_*.out
```

### 3.5 各实验运行方式总览

| 实验 | GPU 需求 | 运行方式 | 预计时间 |
|------|---------|---------|---------|
| 一（分词器） | **无需 GPU** | 登录节点直接运行 | ~10 分钟 |
| 二（模型架构） | **无需 GPU** | 登录节点代码走读 | 4 小时 |
| 三（数据管线） | **无需 GPU** | 登录节点代码走读 | 3 小时 |
| 四（预训练-小） | 1 GPU | `srun --pty` 交互 | ~30 分钟 |
| 四（预训练-标准） | 2 GPU (DDP) | `sbatch` 批量 | ~4-6 小时 |
| 五（优化器） | 1 GPU | `srun --pty` 交互 | 4 小时 |
| 六（SFT） | 2 GPU (DDP) | `sbatch` 批量 | ~2-4 小时 |
| 七（RL） | 2 GPU (DDP) | `sbatch` 批量 | ~4-6 小时 |
| 八（推理/服务） | 1 GPU | `srun --pty` 交互 | 3 小时 |
| 九（评估） | 1 GPU | `srun` 单命令 | 1-2 小时 |
| 十（全流程） | 2 GPU | `sbatch` 批量 | ~8-12 小时 |

### 3.6 实验记录准备

创建个人实验日志文件，记录每次实验的关键参数和观察：

```bash
touch MY_EXPERIMENT_LOG.md
echo "# $(whoami)'s NanoChat Experiment Log" >> MY_EXPERIMENT_LOG.md
echo "Started: $(date)" >> MY_EXPERIMENT_LOG.md
```

---

## 4. 实验一：分词器训练与评估

> **预计时间**：2 小时（含理解 + 实操）
>
> **核心文件**：`scripts/tok_train.py`、`scripts/tok_eval.py`、`nanochat/tokenizer.py`
>
> **技能标签**：#BPE #Tokenization #CompressionMetrics

### 4.1 理论知识

#### 4.1.1 为什么分词很重要？

分词器是 LLM 的"第一层"——它决定了模型看到的"词汇"粒度。一个好的分词器需要平衡：

1. **压缩率**：用尽可能少的 token 表示文本（提升推理速度）
2. **覆盖率**：能正确编码所有可能的输入（避免 `<unk>` token）
3. **语义粒度**：token 边界应与语义边界对齐
4. **多语言支持**：对非英语文本也能合理编码

#### 4.1.2 BPE（Byte-Pair Encoding）原理

BPE 训练过程：
1. 将文本按 UTF-8 字节拆分（确保覆盖所有 Unicode 字符）
2. 统计相邻 token 对的出现频率
3. 合并最高频的 token 对，生成新 token
4. 重复步骤 2-3 直到达到目标词汇量（本项目为 32,768）

#### 4.1.3 NanoChat 分词器的特殊设计

- **词汇量**：32,768（2^15，便于高效索引）
- **正则分割模式**：GPT-4 风格模式，对数字做了特殊处理（`\p{N}{1,2}`，2 位数字分组）
- **特殊 token**：`<|bos|>`（对话开始）、`<|user_start|>` / `<|user_end|>`（用户消息边界）、`<|assistant_start|>` / `<|assistant_end|>`（助手消息边界）、`<|python_start|>` / `<|python_end|>`（工具调用边界）、`<|output_start|>` / `<|output_end|>`（工具输出边界）
- **推理优化**：训练使用 RustBPE（Rust 实现，速度快），推理使用 tiktoken（纯 Python，零依赖，API 兼容 OpenAI）

### 4.2 动手实验

#### Step 1：理解分词器代码

阅读并注释以下文件的关键逻辑：

```bash
# 阅读 BPE 训练流程
cat scripts/tok_train.py

# 阅读分词器封装类
cat nanochat/tokenizer.py
```

**需要回答的问题**：
1. 为什么要分别使用 RustBPE 和 tiktoken 两个库？
2. 特殊 token 的 ID 分配有什么约定？
3. `token_bytes.pt` 文件的作用是什么？

#### Step 2：运行分词器训练

```bash
# 训练分词器（可选 --num_shards 控制数据量）
python -m scripts.tok_train

# 预期输出：
# - tokenizer/ 目录下的 BPE 模型文件
# - tokenizer/token_bytes.pt（每个 token 对应的 UTF-8 字节数）
```

#### Step 3：评估分词器性能

```bash
# 评估分词器压缩率
python -m scripts.tok_eval
```

记录以下指标到你的实验日志：

| 指标 | 你的结果 | 含义 |
|------|---------|------|
| 字符/token 比率 | | 越高压缩越好 |
| 字节/token 比率 | | 越高压缩越好 |
| 词汇覆盖率 | | 100% 最理想 |

#### Step 4：交互式探索

```python
# 在 Python REPL 中探索
from nanochat.tokenizer import get_tokenizer, get_token_bytes

enc = get_tokenizer()

# 测试编码
text = "Hello, world! 你好世界！"
tokens = enc.encode(text)
print(f"原文: {text}")
print(f"Token IDs: {tokens}")
print(f"Token 数: {len(tokens)}")
print(f"每个 token: {[enc.decode([t]) for t in tokens]}")

# 测试特殊 token
print(f"BOS token: {enc.encode('<|bos|>')}")
print(f"User start: {enc.encode('<|user_start|>')}")
```

### 4.3 思考题

1. 如果词汇量从 32K 增大到 64K，对训练和推理分别有什么影响？
2. 为什么 GPT-4 风格的正则模式要对数字做 2 位分组（`\p{N}{1,2}`）而不是按单词切分？
3. 如何评估一个分词器对于中文文本的编码效率？

---

## 5. 实验二：GPT 模型架构深度解析

> **预计时间**：4 小时（含理解 + 代码走读）
>
> **核心文件**：`nanochat/gpt.py`、`nanochat/flash_attention.py`
>
> **技能标签**：#Transformer #GQA #RoPE #QKNorm #FlashAttention

### 5.1 理论知识

#### 5.1.1 Transformer 解码器基础

GPT 模型是自回归 Transformer 解码器，核心公式为：

$$P(x_t | x_{<t}) = \text{softmax}(W_{out} \cdot \text{TransformerBlocks}(W_{emb} \cdot x_{<t}))$$

每个 Transformer 块包含：
1. **多头自注意力**（带因果掩码）：允许每个 token 关注其之前的所有 token
2. **前馈网络**（MLP）：对每个位置独立进行非线性变换

#### 5.1.2 NanoChat 的 14 项架构改进

| # | 改进 | 位置 | 原理与收益 |
|---|------|------|----------|
| 1 | **RoPE**（旋转位置编码） | 注意力 Q/K | 用旋转矩阵编码相对位置，外推性优于绝对位置编码 |
| 2 | **QK 归一化**（RMSNorm） | 注意力前 | 对 Q、K 分别归一化，稳定训练，防止注意力 logits 发散 |
| 3 | **GQA**（分组查询注意力） | 注意力 | KV 头数 < Q 头数，减少 KV Cache 内存，推理加速 |
| 4 | **滑动窗口注意力** | 注意力 | 每层可能限制注意力范围，减少计算量，最后层全注意力 |
| 5 | **ReLU² 激活** | MLP | `relu(x).square()` 替代 GELU，计算更简单，效果相当 |
| 6 | **未绑定权重** | 输出 | token 嵌入和 lm_head 独立，避免共享权重的梯度冲突 |
| 7 | **无偏置** | 全模型 | 所有 Linear 层 `bias=False`，减少参数，简化优化 |
| 8 | **无参数 RMSNorm** | 归一化 | 无可学习 gamma/beta，进一步减少参数 |
| 9 | **嵌入后归一化** | 嵌入层后 | token 嵌入后立即 RMSNorm，稳定梯度 |
| 10 | **值残差**（ResFormer） | 交替注意力层 | 将值嵌入以可学习门控加到 V 上，增强表示能力 |
| 11 | **残差缩放标量** | 每层残差连接 | 可学习的 `resid_lambdas` 缩放每层残差贡献 |
| 12 | **x0 混合标量** | 每层输入 | 可学习的 `x0_lambdas` 混合初始嵌入，类似"高速通道" |
| 13 | **涂抹**（Smear） | 嵌入后 | 混合前一个 token 的嵌入，注入 bigram 先验 |
| 14 | **回溯**（Lookback） | 最终归一化前 | 减去中间层缓存特征，类似"反向残差" |
| 15 | **对数 softcap** | lm_head 后 | `15 * tanh(logits / 15)` 平滑限制 logits，防止极端概率 |

### 5.2 动手实验

#### Step 1：阅读模型定义

```bash
# 完整阅读模型代码（约 400 行）
cat nanochat/gpt.py
```

画出以下结构的计算图：
- 单个 Transformer Block 的数据流
- 注意力计算流程（从 QKV 投影到输出投影）
- RoPE 的施加位置

#### Step 2：构建不同规模的模型

```python
from nanochat.gpt import GPT, GPTConfig, build_model_meta

# 观察不同 depth 下的模型配置
for depth in [6, 12, 20, 24, 32]:
    cfg = build_model_meta(depth, aspect_ratio=64, head_dim=128)
    model = GPT(cfg)
    params = sum(p.numel() for p in model.parameters())
    scaling = model.num_scaling_params()
    print(f"depth={depth}: n_embd={cfg.n_embd}, n_head={cfg.n_head}, "
          f"总参数={params/1e6:.0f}M, 缩放参数={scaling/1e6:.0f}M")
```

记录不同 depth 的参数量变化规律。

#### Step 3：理解 Flash Attention 接口

```bash
cat nanochat/flash_attention.py
```

理解以下设计：
1. 如何根据 GPU 架构（sm90+）自动选择 FA3 或 SDPA？
2. SDPA 回退如何构建滑动窗口掩码？
3. KV Cache 推理时的接口设计

#### Step 4：验证前向计算

```python
import torch
from nanochat.gpt import GPT, GPTConfig

# 构建小模型并测试前向传播
model = GPT(GPTConfig(n_layer=4, n_embd=256, n_head=4, vocab_size=1000))
model.eval()

# 测试训练模式（给定 labels 即计算损失）
x = torch.randint(0, 1000, (2, 128))
y = torch.randint(0, 1000, (2, 128))
loss = model(x, y)
print(f"训练损失: {loss.item()}")

# 测试生成模式
out = model.generate(torch.randint(0, 1000, (1, 10)), max_new_tokens=20, temperature=0.8)
print(f"生成 shape: {out.shape}")
```

#### Step 5：FLOPs 估算分析

```python
# 理解 FLOPs 估算函数
model = GPT(GPTConfig(n_layer=12, n_embd=768, n_head=12, vocab_size=32768))
model.eval()

batch_size = 524288  # 2^19
seq_len = 2048
total_tokens = batch_size * seq_len

flops_per_token = model.estimate_flops()
total_flops = flops_per_token * total_tokens
print(f"每 token FLOPs: {flops_per_token / 1e6:.2f} MFLOPS")
print(f"总 FLOPs: {total_flops / 1e18:.4f} EFlops")
```

### 5.3 思考题

1. 为什么滑动窗口注意力在大部分层使用但最后一层使用全注意力？这背后的直觉是什么？
2. GQA 如何减少推理阶段的 KV Cache 内存？如果 `n_head=32`、`n_kv_head=8`，能节省多少 KV Cache？
3. "无偏置 RMSNorm"（无可学习参数）的优缺点是什么？在什么情况下可能成为瓶颈？
4. RoPE 相比绝对位置编码的数学优势是什么？为什么它能实现长度外推？

---

## 6. 实验三：数据管线与高效数据加载

> **预计时间**：3 小时
>
> **核心文件**：`nanochat/dataset.py`、`nanochat/dataloader.py`
>
> **技能标签**：#DataPipeline #BestFitPacking #DDP #Streaming

### 6.1 理论知识

#### 6.1.1 预训练数据：ClimbMix-400B

NanoChat 使用 NVIDIA 发布的 ClimbMix-400B 数据集（约 400B token），来自 HuggingFace。数据格式为 zstd 压缩的 Parquet 文件，通过流式读取避免一次性加载到内存。

#### 6.1.2 BOS 对齐的 Best-Fit 打包算法

这是 NanoChat 数据加载器的核心创新。目标是将变长文档打包到固定长度（2048 token）的训练序列中，同时满足约束：

1. **BOS 对齐**：每个文档以 `<|bos|>` 开头，确保序列边界与文档边界对齐
2. **Best-Fit 选择**：维护文档缓冲区，每次选择"最大能容纳"的文档填入剩余空间
3. **裁剪填充**：当没有文档能完全放入时，裁剪一个文档以精确填满
4. **100% 利用率**：无填充 token，但约 35% 的文档会被裁剪

**示意图**：

```
序列缓冲区: [BOS|doc1_tokens... 剩余空间: 500 tokens]
文档缓冲区: [docA (800 tokens), docB (300 tokens), docC (600 tokens)]

→ Best-Fit 选择 docB (300 ≤ 500, 是最接近的)
→ 序列缓冲区: [BOS|doc1...|BOS|docB... 剩余空间: 200 tokens]
→ 无文档能放入，裁剪 docA 前 200 tokens 填充
→ 序列完成: 2048 tokens，利用率 100%
```

#### 6.1.3 分布式数据加载

- **分片策略**：每个 rank 读取不同的 Parquet 分片子集（`rank` 起始，`world_size` 步长）
- **确定性打乱**：相同 seed 下各 rank 读取的数据集确定且不重叠
- **状态恢复**：支持 `dataloader_state_dict` 保存/恢复，实现中断续训

### 6.2 动手实验

#### Step 1：下载预训练数据

```bash
# 下载 1 个分片作为实验（约 100MB）
# 完整训练需要更多分片（如 -n 1000）
python -m nanochat.dataset -n 1
```

观察下载后的目录结构。

#### Step 2：理解数据加载代码

```bash
cat nanochat/dataloader.py
```

重点关注：
1. `parquets_iter_batched()` 如何实现流式读取
2. Best-Fit 打包的核心循环
3. DDP 分片逻辑

#### Step 3：测试数据加载器

```python
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state
from nanochat.tokenizer import get_tokenizer

# 创建数据加载器迭代器
enc = get_tokenizer()
data_iter = tokenizing_distributed_data_loader_with_state(
    split="train",
    bos_token_id=enc.bos_token_id,
    device_batch_size=4,
    max_seq_len=256,
    device="cpu",
    resume_state_dict=None,
)

# 查看一批数据
x, y, dataloader_state_dict = next(data_iter)
print(f"输入 shape: {x.shape}")  # (4, 256)
print(f"标签 shape: {y.shape}")  # (4, 256)
print(f"输入开头 token: {x[0, :10]}")
print(f"对应的标签: {y[0, :10]}")  # y 是 x 右移一位
```

#### Step 4：分析打包效率

修改代码，统计以下指标并记录：
- 文档完整保留率（未被裁剪的比例）
- 平均每序列包含的文档数
- 序列填充率（≈100%）

### 6.3 思考题

1. 相比"简单拼接后切块"方案，BOS 对齐的 Best-Fit 打包有什么优势？
2. 为什么 35% 的裁剪率是可以接受的？这对学习文档边界的模型有什么影响？
3. 如果上下文长度从 2048 扩展到 8192，对打包效率和训练效率分别有什么影响？

---

## 7. 实验四：基础模型预训练

> **预计时间**：4 小时（代码走读为主，训练可选运行）
>
> **核心文件**：`scripts/base_train.py`
>
> **技能标签**：#Pretraining #ScalingLaws #LearningRateSchedule #GradientAccumulation

### 7.1 理论知识

#### 7.1.1 Chinchilla 缩放定律

本项目遵循 Chinchilla 缩放定律的核心结论：**计算最优训练要求训练 token 数与模型参数近似成正比（约 20:1）**。

NanoChat 的实现公式：

- `scaling_params` = transformer 矩阵参数 + lm_head 参数（排除嵌入，遵循 Kaplan 风格）
- `target_tokens` = `scaling_params × target_param_data_ratio`（默认 ratio=12）
- `num_iterations` = `target_tokens / total_batch_size`

#### 7.1.2 自动超参数推导

以 depth=12 为参考（批次大小 2^19 = 524,288），更大模型通过幂律自动推导：

| 参数 | 公式 | 原理 |
|------|------|------|
| 批次大小 | $B \propto D^{0.383}$ | 更大模型需要更大批次以维持梯度信噪比 |
| 学习率 (AdamW) | $LR \propto \sqrt{B / B_{ref}}$ | 批次增大时学习率需相应提升 |
| 学习率 (Muon) | 同上 | 相同缩放规律 |
| 权重衰减 | 保持 $B / (LR \cdot WD \cdot D)$ 恒定 | T_epoch 框架约束 |

#### 7.1.3 训练循环详解

```
for step in range(num_iterations):
    1. 累积 micro-batch 梯度 (grad_accum_steps 次)
    2. loss.backward()
    3. optimizer.step() + optimizer.zero_grad()
    4. 按调度更新学习率
    5. 定期评估（BPB、CORE、采样）
    6. 定期保存检查点
```

**梯度累积**：`grad_accum_steps = total_batch_size / device_batch_size`
- 设备批次 32 × 梯度累积 256 × 2 GPU（DDP 每个 rank）= 总批次 16,384（可随 depth 自动调整）

#### 7.1.4 学习率调度

采用三阶段调度：

```
预热期 (40 steps):        lr: 0 → initial
平稳期 (~35% steps):       lr: initial (恒定)
冷却期 (~65% steps):       lr: initial → 0.05×initial (线性)
```

### 7.2 动手实验

#### Step 1：阅读训练脚本

```bash
cat scripts/base_train.py
```

重点关注：
1. `build_model_meta()` 如何根据 depth 自动配置模型
2. 缩放定律如何计算 `num_iterations`
3. 学习率调度器的实现
4. 评估点的设计（BPB / CORE / 采样 / 检查点）

#### Step 2：分析缩放定律

```python
import math

def estimate_training_config(depth: int, param_data_ratio: float = 12):
    """分析不同 depth 下的训练配置"""
    aspect_ratio = 64
    head_dim = 128
    n_embd = round(depth * aspect_ratio / head_dim) * head_dim
    n_head = n_embd // head_dim

    # 参考 d12
    d12_dim = 768
    d12_scaling = 84_934_144  # d12 的缩放参数

    # 估算当前模型的缩放参数比例
    # scaling ∝ dim^2 × depth
    dim_ratio = n_embd / d12_dim
    scaling_ratio = (dim_ratio ** 2) * (depth / 12)

    batch_size = 2**19 * (scaling_ratio ** 0.383)
    batch_size = 2**round(math.log2(batch_size))

    lr_muon = 0.02 * math.sqrt(batch_size / 2**19)

    return {
        'n_embd': n_embd,
        'n_head': n_head,
        'batch_size': int(batch_size),
        'lr_muon': round(lr_muon, 4),
        'target_tokens_B': round(scaling_ratio * d12_scaling * param_data_ratio / 1e9, 1),
    }

for d in [12, 16, 20, 24, 28, 32]:
    cfg = estimate_training_config(d)
    print(f"depth={d:2d}: dim={cfg['n_embd']}, heads={cfg['n_head']}, "
          f"batch={cfg['batch_size']}, lr_muon={cfg['lr_muon']}, "
          f"tokens={cfg['target_tokens_B']}B")
```

#### Step 3：运行小型训练（CPU 演示模式）

```bash
# 在 CPU 上运行一个极小的实验（depth=6，约 30 分钟）
bash runs/runcpu.sh

# 或手动指定参数
python -m scripts.base_train --depth=6 --max-seq-len=256 \
    --device-batch-size=8 --num-iterations=500 \
    --run=demo_experiment
```

#### Step 4：监控训练过程

如果使用 WandB，观察以下曲线：
- 训练损失下降曲线
- BPB 评估曲线
- 学习率变化曲线
- 梯度范数变化

记录最终结果到实验日志：

```
| depth | 总步数 | 最终 loss | BPB | CORE | 训练时间 | GPU 数量 |
|-------|--------|----------|-----|------|---------|---------|
| 6     | 500    |        |     |      |         | CPU     |
```

### 7.3 思考题

1. 为什么参考 depth 设为 12，而不是更小或更大的值？
2. 梯度累积和增大批次大小对训练稳定性有什么不同的影响？
3. 预热期为什么重要？没有预热期会发生什么？
4. Chinchilla 定律给出了 20:1 的 token:param 比，为什么 NanoChat 默认用 12:1？

---

## 8. 实验五：混合优化器与分布式训练

> **预计时间**：4 小时
>
> **核心文件**：`nanochat/optim.py`、`nanochat/fp8.py`、`nanochat/common.py`
>
> **技能标签**：#Muon #AdamW #ZeRO #FP8 #DDP #OptimizerDesign

### 8.1 理论知识

#### 8.1.1 为什么混合优化器？

传统做法是对所有参数使用 AdamW，但研究发现：

- **矩阵参数**（权重矩阵）：适合用 Muon（基于动量正交化），因为它直接优化矩阵的奇异值分布
- **向量/标量参数**（嵌入、bias、增益标量）：适合用 AdamW，因为它们结构简单，自适应学习率更稳定

NanoChat 的划分逻辑（`setup_optimizer()`）：

| 参数类型 | 优化器 | 学习率 | 特殊设置 |
|---------|--------|--------|---------|
| Transformer 矩阵权重 | Muon | 0.02 | momentum=0.95, ns_steps=5 |
| Token 嵌入 | AdamW | 0.2 | betas=(0.8, 0.995) |
| lm_head | AdamW | 0.004 | betas=(0.8, 0.96) |
| 值嵌入 (ResFormer) | AdamW | 0.1 | — |
| 残差标量 (resid_lambdas) | AdamW | 0.005 | — |
| x0 标量 (x0_lambdas) | AdamW | 0.5 | beta1=0.96 |
| 涂抹/回溯标量 | AdamW | 0.2 | — |

#### 8.1.2 Muon 优化器原理

Muon（MomentUm Orthogonalized by Newton-schulz）的核心步骤：

1. **Nesterov 动量**：velocity = momentum × velocity + grad（在插值后的位置评估）
2. **Polar Express 正交化**：通过 Newton-Schulz 迭代 ($Y' = Y \cdot (3I - Y^T Y) / 2$)，5 次迭代将矩阵推向正交流形
3. **NorMuon 方差缩减**：按列/行的自适应学习率缩放
4. **谨慎权重衰减**：只对"离开零太远"的参数施加惩罚

#### 8.1.3 ZeRO-2 风格优化器分片

`DistMuonAdamW` 实现了三阶段异步通信：

```
Phase 1: 发起所有 reduce_scatter (大参数) / all_reduce (小参数)
Phase 2: 等待完成 → 计算更新 → 发起 all_gather
Phase 3: 等待完成 → 复制参数
```

与 DDP 原生的区别：不使用 `nn.parallel.DistributedDataParallel`，梯度同步由优化器自行处理。

#### 8.1.4 FP8 混合精度训练

FP8 分为两种格式：
- **E4M3**（4 位指数 + 3 位尾数）：高精度，用于前向传播
- **E5M2**（5 位指数 + 2 位尾数）：宽范围，用于反向传播

`Float8Linear` 的实现（`nanochat/fp8.py`，约 150 行）：
- 使用 `torch._scaled_mm`（cuBLAS FP8 矩阵乘法内核）
- 张量级动态缩放（每张量一个缩放因子）

### 8.2 动手实验

#### Step 1：阅读优化器代码

```bash
# 完整阅读 Muon + AdamW 实现
cat nanochat/optim.py

# 阅读 FP8 实现
cat nanochat/fp8.py
```

画出以下流程图：
1. Muon 步的完整计算图（Nesterov → Polar Express → NorMuon → Weight Decay）
2. DistMuonAdamW 的三阶段通信模式

#### Step 2：理解参数分组

```python
from nanochat.gpt import GPT, GPTConfig

model = GPT(GPTConfig(n_layer=4, n_embd=256, n_head=4, vocab_size=1000))

# 观察 setup_optimizer 如何分组
param_groups = model.setup_optimizer(
    device_type="cuda",
    adamw_lr=0.2,
    adamw_betas=(0.8, 0.995),
    muon_lr=0.02,
    muon_momentum=0.95,
)

for i, pg in enumerate(param_groups):
    shapes = [p.shape for p in pg['params']]
    print(f"Group {i}: optimizer={pg.get('optimizer','?')}, lr={pg['lr']:.4f}, "
          f"params={len(shapes)}, shapes={shapes[:3]}...")
```

#### Step 3：分析 Muon 正交化的效果

```python
import torch

# 模拟一个随机梯度矩阵
torch.manual_seed(42)
W = torch.randn(128, 256)
grad = torch.randn(128, 256)

# 计算 Muon 更新方向（简化版）
# Newton-Schulz 迭代
def newton_schulz(X, steps=5):
    # 使 X 尽可能接近正交矩阵
    for _ in range(steps):
        X = X @ (3 * torch.eye(X.shape[1]) - X.T @ X) / 2
    return X

# 对比原始梯度和正交化后的梯度
orig_direction = grad
muon_direction = newton_schulz(grad)

# 检查奇异值分布
_, S_orig, _ = torch.svd(orig_direction)
_, S_muon, _ = torch.svd(muon_direction)

print(f"原始梯度奇异值: min={S_orig.min():.4f}, max={S_orig.max():.4f}, std={S_orig.std():.4f}")
print(f"Muon 梯度奇异值: min={S_muon.min():.4f}, max={S_muon.max():.4f}, std={S_muon.std():.4f}")
# Muon 使奇异值更均匀，避免某些方向更新过快/过慢
```

#### Step 4：FP8 转换实验（需 H100 GPU）

```bash
python -m scripts.base_train --depth=12 --fp8 --num-iterations=100 --run=fp8_test
```

对比 `--fp8` 和默认 (BF16) 的训练速度和内存占用的差异。

### 8.3 思考题

1. Muon 中的 Newton-Schulz 迭代为什么是必要的？如果直接用原始梯度有什么问题？
2. ZeRO-2 分片和 DDP 全复制之间的通信量差异是多少？在什么规模下分片的收益最大？
3. FP8 的 4 位指数在什么情况下会溢出？如何处理这种溢出？
4. 为什么 lm_head 的学习率设为 0.004（远小于嵌入的 0.2）？这背后的直觉是什么？

---

## 9. 实验六：监督微调（SFT）

> **预计时间**：3 小时
>
> **核心文件**：`scripts/chat_sft.py`、`tasks/` 目录
>
> **技能标签**：#SupervisedFineTuning #SFT #MultiTaskLearning #ConversationFormat

### 9.1 理论知识

#### 9.1.1 预训练 vs 微调

| 维度 | 预训练 | 监督微调 |
|------|--------|---------|
| 目标 | 学习语言统计规律（下一个 token 预测） | 学习对话格式和任务指令 |
| 数据格式 | 纯文本序列 | 多轮对话（role + content） |
| 数据量 | 极大（百亿 token 级） | 较小（百万条级） |
| 损失计算 | 所有 token | 仅 assistant token（用户 token 被掩码） |
| 学习率 | 较高 | 约为预训练的 80% |

#### 9.1.2 对话格式

NanoChat 使用特殊 token 明确标记对话结构：

```
<|bos|><|user_start|>你好，请帮我写一个排序函数<|user_end|>
<|assistant_start|>当然，以下是一个 Python 排序函数...<|assistant_end|>
<|user_start|>能优化一下吗？<|user_end|>
<|assistant_start|>好的，可以有这些优化...<|assistant_end|>
```

对于包含工具调用的数学题（GSM8K）：

```
<|user_start|>小明有 5 个苹果，买了 3 个，吃了 2 个，还剩几个？<|user_end|>
<|assistant_start|>让我们一步步计算：
<|python_start|>5 + 3 - 2<|python_end|><|output_start|>6<|output_end|>
所以还剩 6 个苹果。<|assistant_end|>
```

#### 9.1.3 损失掩码策略

仅对 assistant 消息中的 token 计算损失，掩码规则：
- ✅ `<|assistant_start|>` 到 `<|assistant_end|>` 之间的 token
- ❌ `<|bos|>` token
- ❌ `<|user_start|>` 到 `<|user_end|>` 之间的 token（用户输入）
- ❌ `<|output_start|>` 到 `<|output_end|>` 之间的 token（工具输出，是给定的）

#### 9.1.4 多任务混合数据

SFT 阶段混合了多种任务的数据：

| 任务 | 数据量 | 周期 | 作用 |
|------|--------|------|------|
| SmolTalk | 460K | 1 | 通用对话能力 |
| CustomJSON (身份) | 1K | 1 | 模型身份/安全训练 |
| MMLU | 100K | 3 | 知识问答 |
| GSM8K | 8K | 4 | 数学推理 |
| SimpleSpelling | 200K | 1 | 字符级理解 |
| SpellingBee | 80K | 1 | 计数/拼写 |

### 9.2 动手实验

#### Step 1：理解任务基类

```bash
cat tasks/common.py    # Task 和 TaskMixture 基类
cat tasks/smoltalk.py  # 通用对话任务
cat tasks/mmlu.py      # 知识问答任务
cat tasks/gsm8k.py     # 数学推理任务
```

#### Step 2：分析对话数据格式

```python
from tasks.smoltalk import SmolTalk
from nanochat.tokenizer import get_tokenizer

task = SmolTalk(split="train")
example = task.get_example(0)
print(example)

enc = get_tokenizer()
tokens = enc.render_conversation(example)
decoded = enc.decode(tokens)
print(f"\nToken 数量: {len(tokens)}")
print(f"解码后:\n{decoded}")
```

#### Step 3：理解损失掩码

```python
from tasks.common import render_data_batch

# 手动构造一个带掩码的批次
batch = [task.get_example(i) for i in range(4)]
x, y, mask = render_data_batch(batch, enc)

# 分析 mask 分布
total_tokens = mask.numel()
assistant_tokens = mask.sum().item()
print(f"总 token: {total_tokens}")
print(f"有损失的 token: {assistant_tokens} ({100*assistant_tokens/total_tokens:.1f}%)")
```

#### Step 4：运行 SFT 训练

```bash
# 基于预训练模型进行 SFT
python -m scripts.chat_sft --depth=12 --run=sft_experiment \
    --num-iterations=500
```

记录 SFT 前后的模型行为变化。

### 9.3 思考题

1. 为什么只对 assistant token 计算损失？如果对所有 token 计算损失会有什么问题？
2. MMLU 和 GSM8K 的数据重复了多个周期（3 周期/4 周期），为什么不把它们和 SmolTalk 一样只过一遍？
3. 为什么 SFT 的学习率选为预训练的 80%？过高或过低会有什么问题？

---

## 10. 实验七：强化学习（RLHF/GRPO）

> **预计时间**：4 小时
>
> **核心文件**：`scripts/chat_rl.py`、`nanochat/engine.py`
>
> **技能标签**：#GRPO #ReinforcementLearning #PreferenceOptimization #ExplorationSampling

### 10.1 理论知识

#### 10.1.1 从 SFT 到 RL

SFT 训练模型"模仿"好的回答，但存在局限：
- 没有"好/坏"的对比信号——模型不知道自己的输出质量
- 无法利用"试错"学习——每次都从正确答案学习，没有探索过程

强化学习（RL）补足了这个缺陷：让模型生成多个样本，根据奖励信号（reward）区分好坏，然后鼓励"好"的生成模式。

#### 10.1.2 GRPO（Group Relative Policy Optimization）

GRPO 是 DeepSeek-R1 使用的算法。与 PPO 的核心区别：

| 特性 | PPO | GRPO |
|------|-----|------|
| 需要参考模型 | 是（计算 KL） | 否 |
| KL 正则化 | 需要 | 不需要 |
| 优势计算 | GAE（时序差分） | 组内均值减法 |
| 裁剪机制 | 重要性采样比率裁剪 | 无裁剪 |

NanoChat 的简化版 GRPO：

1. **采样**：对每个问题，用当前模型生成 K 个回答（默认 16 个）
2. **评分**：每个回答获得奖励（GSM8K 中：答案匹配 = 1，否则 = 0）
3. **优势计算**：`advantage = reward - mean(rewards)`（组内均值减法）
4. **策略梯度**：`objective = mean(log_prob(token) × advantage)`
5. **无 KL 惩罚、无裁剪**——极简设计，仅靠均值减法防止梯度爆炸

#### 10.1.3 为什么在数学推理上做 RL？

GSM8K 数学题的优势：
- **奖励信号明确**：答案对/错，无歧义
- **需要探索**：同一问题有多种解法，模型可尝试不同推理路径
- **验证容易**：通过工具调用（Python 计算器）可自动验证

### 10.2 动手实验

#### Step 1：阅读 RL 训练脚本

```bash
cat scripts/chat_rl.py
```

重点理解：
1. 采样循环（如何用 Engine 生成多个样本）
2. 奖励计算（如何解析 GSM8K 答案）
3. 优势计算（组内相对分数）
4. 策略梯度目标

#### Step 2：理解奖励计算

```python
from tasks.gsm8k import GSM8K

task = GSM8K(split="train")
example = task.get_example(0)
print(f"问题: {example['messages'][0]['content']}")
print(f"参考答案: {example['messages'][1]['content']}")

# GSM8K 的奖励逻辑
ground_truth = task.evaluate(example)  # 正确答案
model_answer = "42"  # 模型生成的答案

# reward = 1 if model_answer == ground_truth else 0
```

#### Step 3：分析 GRPO 策略梯度

```python
import torch

# 模拟 GRPO 训练的一步
K = 16  # 每组样本数
vocab_size = 1000
seq_len = 100

# 模拟 16 个样本的 log 概率和奖励
log_probs = torch.randn(K, seq_len)  # 每个 token 的 log prob
rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

# 组内均值作为基线
baseline = rewards.mean()
advantages = rewards - baseline
print(f"奖励分布: {rewards.tolist()}")
print(f"基线 (均值): {baseline:.3f}")
print(f"优势: {advantages.tolist()}")
print(f"\n正优势样本: {(advantages > 0).sum().item()}")
print(f"负优势样本: {(advantages < 0).sum().item()}")

# 策略梯度损失
# loss = -mean(log_prob * advantage)
policy_loss = -(log_probs * advantages.unsqueeze(-1)).mean()
print(f"策略梯度损失: {policy_loss.item():.4f}")
```

#### Step 4：运行 RL 训练

```bash
# 在 SFT 模型基础上进行 RL 训练
python -m scripts.chat_rl --depth=12 --run=rl_experiment
```

记录以下指标：
- RL 训练的损失变化
- pass@1、pass@4、pass@16 的变化
- 样本多样性的变化

### 10.3 思考题

1. 为什么 NanoChat 选择不实现 KL 惩罚？这在什么条件下是合理的？什么情况下可能需要加回 KL 惩罚？
2. 组内均值减法（而非绝对奖励阈值）的优势和局限性是什么？如果所有样本都正确或都错误会怎样？
3. 为什么选择在 GSM8K（数学推理）上做 RL，而不是在通用对话上？如果换成通用对话，奖励信号如何设计？

---

## 11. 实验八：推理引擎与模型服务化

> **预计时间**：4 小时
>
> **核心文件**：`nanochat/engine.py`、`scripts/chat_web.py`、`scripts/chat_cli.py`、`nanochat/ui.html`
>
> **技能标签**：#InferenceEngine #KVCache #Streaming #FastAPI #ModelDeployment

### 11.1 理论知识

#### 11.1.1 自回归推理的两阶段

解码过程可分为两个阶段：

1. **预填充（Prefill）**：一次性处理整个提示序列
   - 计算所有层的 Key/Value 并存入 KV Cache
   - 计算密集型（矩阵乘），内存开销可控
   
2. **自回归解码（Decode）**：逐 token 生成
   - 每次仅处理一个 token 的 Q、K、V
   - 内存密集型（受 KV Cache 访问速度限制）

#### 11.1.2 KV Cache 管理

KV Cache 是推理优化的核心。NanoChat 的实现：

- **布局**：`(B, max_seq_len, n_kv_head, head_dim)` —— 与 Flash Attention 3 的要求对齐
- **GQA 节省**：KV 头数少，Cache 占用更少（例如 n_head=32, n_kv_head=8 → 节省 75%）
- **多采样共享**：预填充后，KV Cache 被复制用于 K 个独立样本的解码

#### 11.1.3 流式推理

服务端通过 Server-Sent Events (SSE) 实现流式输出：

```
Client                    Server
  |                          |
  |-- POST /chat/completions |
  |                          |-- 预填充
  |                          |-- 生成 token 1
  |<-- data: {"token": "你"} |
  |                          |-- 生成 token 2
  |<-- data: {"token": "好"} |
  |                          |-- 生成 token 3
  |<-- data: {"token": "！"} |
  |<-- data: [DONE]          |
```

**UTF-8 累积解码**：由于中文字符等多字节字符可能被拆分为多个 token，服务端需要累积 token 直到形成完整的 UTF-8 字符才能发送给客户端。

#### 11.1.4 多 GPU 工作池

`scripts/chat_web.py` 实现了一个简单的工作池模式：
- 每个 GPU 加载一份模型副本
- 请求到达时，分配到可用工作器
- 若所有工作器忙，请求排队等待
- 通过 `/health` 和 `/stats` 端点监控

### 11.2 动手实验

#### Step 1：阅读推理引擎

```bash
cat nanochat/engine.py
```

重点关注：
1. KV Cache 的预分配和布局
2. `prefill()` 和 `generate_batch()` 的实现
3. 工具调用（Python 计算器）的检测和执行
4. 多采样的实现

#### Step 2：启动命令行聊天

```bash
# 启动交互式聊天
python -m scripts.chat_cli --depth=12

# 提问：
# 1. "你好，请介绍一下你自己"
# 2. "计算 123 + 456 的结果"  （观察工具调用）
# 3. "写一个 Python hello world 程序"
```

#### Step 3：启动 Web 服务

```bash
# 启动 Web 服务器（默认端口 8000）
python -m scripts.chat_web --depth=12

# 访问 http://localhost:8000 查看 ChatGPT 风格 UI
# 访问 http://localhost:8000/health 查看健康状态
# 访问 http://localhost:8000/stats 查看工作池状态
```

#### Step 4：分析推理性能

```python
from nanochat.engine import Engine
from nanochat.gpt import GPT, GPTConfig
import time

# 构建模型和引擎
model = GPT(GPTConfig(n_layer=12, n_embd=768, n_head=12, vocab_size=32768))
engine = Engine(model, device="cuda" if torch.cuda.is_available() else "cpu")

# 基准测试
prompt = torch.tensor([[1, 2, 3, 4, 5]])  # 示例 prompt

# 测量预填充时间
t0 = time.time()
engine.prefill(prompt)
t_prefill = time.time() - t0

# 测量解码速度
num_tokens = 100
t0 = time.time()
tokens = engine.generate_batch(num_tokens=num_tokens)
t_decode = time.time() - t0

print(f"预填充: {t_prefill*1000:.1f}ms")
print(f"解码速度: {num_tokens/t_decode:.1f} tokens/s")
```

### 11.3 思考题

1. KV Cache 为什么能加速推理？它用空间换时间的具体机制是什么？
2. 预填充阶段为什么比逐 token 解码快得多（每个 token 的 GPU 利用率）？
3. 如果上下文长度从 2048 扩展到 1M token，KV Cache 会成为什么瓶颈？有哪些解决方案？

---

## 12. 实验九：模型评估体系

> **预计时间**：4 小时
>
> **核心文件**：`nanochat/core_eval.py`、`nanochat/loss_eval.py`、`scripts/chat_eval.py`、`scripts/base_eval.py`、`tasks/`
>
> **技能标签**：#ModelEvaluation #Benchmarking #BPB #CORE #PassK

### 12.1 理论知识

#### 12.1.1 评估维度全景

NanoChat 的评估体系覆盖三个维度：

```
评估维度
├── 语言建模能力
│   ├── BPB (Bits Per Byte) —— 压缩效率，与词汇量无关
│   └── 验证损失 —— 通用语言建模质量
│
├── 知识/推理能力
│   ├── CORE (22 项任务复合指标) —— 基础模型综合能力
│   ├── ChatCORE —— 对话模型综合能力
│   ├── MMLU (57 个主题) —— 多领域知识
│   ├── ARC-Easy / ARC-Challenge —— 科学推理
│   └── GSM8K —— 数学推理
│
└── 代码生成能力
    └── HumanEval —— 代码生成正确率
```

#### 12.1.2 BPB（Bits Per Byte）详解

BPB 是与词汇量无关的评估指标：

$$\text{BPB} = \frac{\sum \text{NLL}_i}{\ln(2) \cdot \sum \text{bytes}_i}$$

- `NLL`：每个 token 的负对数似然
- `bytes`：每个 token 对应的原始 UTF-8 字节数（来自 `token_bytes.pt`）

**优点**：不同词汇量的模型可以直接比较（而困惑度 PPL 不能）

#### 12.1.3 CORE 基准

CORE 来自 DCLM 论文，包含 22 个多样化任务：

| 类型 | 任务数 | 评估方法 |
|------|--------|---------|
| 多项选择 | ~15 | 比较各选项的平均损失，选最低的 |
| 模式匹配 | ~5 | 比较不同上下文的损失 |
| 语言建模 | ~2 | 检查模型是否完美预测了延续 |

结果经过随机基线中心化（0 = 随机水平，1 = 完美），然后取平均。

#### 12.1.4 Pass@k 评估

生成式任务（GSM8K、HumanEval）使用 Pass@k：
- 对每个问题，采样 k 个答案
- 如果任何 1 个通过测试，该问题算通过
- `pass@k = 通过数 / 总问题数`
- 更高 k 意味着更多的尝试机会，更高通过率

### 12.2 动手实验

#### Step 1：基础模型评估

```bash
# 评估基础模型
python -m scripts.base_eval --depth=12
```

理解以下输出：
- CORE 分数及其含义
- BPB 值
- 各个子任务的得分

#### Step 2：对话模型评估

```bash
# 评估 SFT/RL 后的对话模型
python -m scripts.chat_eval --depth=12
```

记录 ChatCORE 各子任务的得分。

#### Step 3：运行 CORE 评估逻辑

```python
from nanochat.core_eval import evaluate_model

# 注意：需要先下载 eval_bundle
# 这个评估比较耗时，了解原理即可
```

#### Step 4：对比评估结果

创建一个评估对照表：

| 基准 | GPT-2 (2019) | NanoChat d12 | NanoChat d20 | NanoChat d24 |
|------|-------------|-------------|-------------|-------------|
| CORE | 0.2565 | - | - | ~0.26 |
| BPB (val) | - | - | - | ~0.72 |
| MMLU | - | - | - | - |
| GSM8K pass@1 | - | - | - | - |

### 12.3 思考题

1. 为什么 BPB 比 PPL（困惑度）更适合作为跨词汇量的比较指标？
2. CORE 评估中对多项选择题使用"最低损失选项"而非"最高概率选项"，两者有什么区别？
3. Pass@k 和准确率（k=1）的区别反映了模型的什么能力？

---

## 13. 实验十：全流程整合与报告生成

> **预计时间**：4 小时
>
> **核心文件**：`scripts/base_train.py`、`scripts/chat_sft.py`、`scripts/chat_rl.py`、`nanochat/report.py`
>
> **技能标签**：#Integration #Reproducibility #ExperimentManagement #TechnicalWriting

### 13.1 理论知识

#### 13.1.1 完整训练管线

```
                    ┌──────────────┐
                    │  数据下载     │
                    │ ClimbMix-400B│
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  分词器训练   │
                    │  BPE 32K vocab│
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  预训练      │
                    │  base_train  │
                    │  (计算最优)   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  基础评估    │
                    │  CORE + BPB  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  监督微调    │
                    │  chat_sft    │
                    │  (多任务混合) │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  强化学习    │
                    │  chat_rl     │
                    │  (GRPO)      │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  最终评估    │
                    │  ChatCORE    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  部署 & 报告 │
                    │  chat_web    │
                    │  report      │
                    └──────────────┘
```

#### 13.1.2 实验记录的最佳实践

- **可复现性**：记录所有超参数、随机种子、数据版本、代码 commit hash
- **对比性**：改变单一变量，保持其他条件不变
- **日志完整**：训练损失、评估指标、系统资源使用（GPU 利用率、内存、时间）
- **结论清晰**：每个实验应有一个明确的结论（"X 改进使 Y 提升了 Z"）

### 13.2 动手实验

#### Step 1：理解报告系统

```bash
cat nanochat/report.py
```

了解报告生成的结构和信息来源。

#### Step 2：运行 Speedrun（完整流程）

```bash
# 阅读 speedrun 脚本
cat runs/speedrun.sh

# 如果资源允许，运行完整流程
bash runs/speedrun.sh
```

#### Step 3：撰写实验报告

基于你的实验数据，按以下模板撰写实验报告：

```markdown
# NanoChat 实习实验报告

## 1. 实验环境
- GPU 型号与数量：___
- CUDA 版本：___
- PyTorch 版本：___
- 实验日期：___

## 2. 分词器训练
- 词汇量：32,768
- 字符/token 比率：___
- 字节/token 比率：___
- 发现与思考：___

## 3. 预训练
| depth | 总参数 | 缩放参数 | 批次大小 | 总步数 | 最终 Loss | BPB | CORE | 训练时间 |
|-------|--------|---------|---------|--------|----------|-----|------|---------|
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

- 缩放定律验证：___
- 损失曲线分析：___

## 4. 监督微调
- SFT 前后 CORE 变化：___
- 对话质量定性评估：___
- 工具调用能力：___

## 5. 强化学习
- GSM8K pass@1 变化：___
- RL 训练稳定性：___
- 样本多样性变化：___

## 6. 推理部署
- 解码速度 (tokens/s)：___
- 显存占用：___
- Web 服务延迟 (P50/P95)：___

## 7. 总结与收获
- 核心技能提升：___
- 遇到的挑战与解决：___
- 对 LLM 工程的理解：___
```

### 13.3 思考题

1. 在完整管线中，如果某个阶段的结果不理想（如 BPB 过高），你会回溯到哪个环节排查？排查思路是什么？
2. 为什么实验记录对 AI 工程师如此重要？这与普通软件开发有什么区别？
3. 如果要向面试官展示这个项目，你会重点强调哪些数据和结论？

---

## 14. 能力图谱与求职映射

### 14.1 技能-实验-岗位 三维映射

| 技能领域 | 对应实验 | 覆盖知识点 | 对口岗位 |
|---------|---------|-----------|---------|
| **模型架构设计** | 实验二 | Transformer 改进、GQA、RoPE、滑动窗口 | 模型算法工程师 |
| **训练系统** | 实验四、五 | 缩放定律、LR 调度、DDP、ZeRO-2 | 大模型训练工程师 |
| **数值精度优化** | 实验五 | FP8、混合精度、GradScaler | AI Infra 工程师 |
| **优化器设计** | 实验五 | Muon、AdamW、动量正交化 | AI 框架开发工程师 |
| **分词系统** | 实验一 | BPE、正则分词、推理优化 | NLP 算法工程师 |
| **数据工程** | 实验三 | Best-Fit 打包、流式处理、DDP 分片 | 数据工程师 |
| **监督微调** | 实验六 | SFT、多任务学习、损失掩码 | 模型微调工程师 |
| **强化学习对齐** | 实验七 | GRPO、策略梯度、奖励设计 | RLHF 算法工程师 |
| **推理引擎** | 实验八 | KV Cache、流式推理、GQA | 推理优化工程师 |
| **模型部署** | 实验八 | FastAPI、SSE、负载均衡 | MLOps 工程师 |
| **模型评估** | 实验九 | CORE、BPB、Pass@k | 模型评测工程师 |
| **工程实践** | 实验十 | 实验管理、可复现性、成本估算 | 技术负责人 |

### 14.2 面试话术建议

#### 自我介绍模板（30 秒电梯演讲）

> "我独立完成了 NanoChat 全栈 LLM 训练项目的学习和实践。这是一个覆盖分词器、预训练、SFT、RL 到推理部署的端到端项目。我深入理解了 Transformer 架构的 14 项现代改进，实践了 Muon 混合优化器、FP8 训练加速、Chinchilla 缩放定律自动超参数推导等技术。我训练的 d20 模型在 CORE 基准上达到了 X.X 分，验证了计算最优训练的理论。"

#### 技术深度问题示例

**Q: 你们项目用 Muon 优化器，它和 AdamW 有什么区别？**
> "Muon 通过 Newton-Schulz 迭代将梯度矩阵推向正交流形，使各奇异值方向的学习速率更均衡。这解决了 AdamW 对矩阵参数优化效率低的问题——AdamW 的自适应学习率对矩阵的奇异值分布不感知。我们项目中，Muon 用于所有 Transformer 权重矩阵，AdamW 用于嵌入和标量参数，两者混合使用。" 

**Q: 为什么选择 GRPO 而不是 PPO？**
> "GRPO 是 DeepSeek-R1 使用的算法，相比 PPO 有几个优势：不需要参考模型（节省一半显存），不需要 KL 惩罚计算，直接在组内通过均值减法计算优势。我们的简化版实现进一步省去了 ratio clipping，仅靠组内相对比较来提供学习信号。在数学推理场景下，因为奖励信号是二元的（对/错），这种方法效果好且实现简单。"

**Q: 你们的 FP8 训练是怎么实现的？**
> "我们约 150 行代码实现了 Float8Linear，替代 torchao。使用 torch._scaled_mm 调用 cuBLAS FP8 内核，前向用 E4M3 格式（高精度），反向用 E5M2 格式（宽范围），每张量动态缩放。主要技巧是用 torch._dynamo.allow_in_graph 防止 torch.compile 分解自定义 autograd 函数。只有 dim≥128 且可被 16 整除的线性层才转换，保证数值稳定。"

### 14.3 简历项目描述参考

```
NanoChat 全栈 LLM 训练框架实践                           2026.03 - 2026.06

• 深入研读并实践了 Karpathy 设计的端到端 LLM 训练框架（约 10K 行 Python）
• 掌握了 GPT 模型的 14 项现代改进（RoPE、GQA、QK Norm、ReLU²、值残差、滑动窗口等）
• 实践了 Chinchilla 缩放定律下的计算最优训练，通过单一 depth 参数自动推导全部超参数
• 深入理解 Muon（动量正交化）+ AdamW 混合优化器、ZeRO-2 优化器状态分片
• 实践了 FP8 混合精度训练（基于 torch._scaled_mm 的 Float8Linear）
• 掌握了 GRPO 强化学习对齐算法（组内相对优势，无需 KL 惩罚）
• 实现了 FastAPI + SSE 流式推理服务，含 KV Cache 管理与多 GPU 工作池
• 掌握 CORE/BPB/ChatCORE/Pass@k 多维度模型评估体系
• 8×H100 上 2 小时训练出超越 GPT-2 水平的对话模型，成本约 15-48 美元

技术栈：PyTorch 2.9, Flash Attention 3, DDP, FastAPI, RustBPE, tiktoken, WandB
```

---

## 15. 评估标准

### 15.1 评分维度

| 评分项 | 权重 | 优秀 (90-100) | 良好 (75-89) | 合格 (60-74) | 不合格 (<60) |
|--------|------|--------------|-------------|------------|------------|
| **代码理解** | 25% | 能清晰解释所有核心文件的逐行逻辑，能指出设计权衡 | 能解释主要流程和关键函数 | 知道每个文件做什么，但不深入 | 无法解释基本代码结构 |
| **实验执行** | 25% | 独立完成所有实验，有系统性的参数探索和对比分析 | 完成核心实验，有基本数据记录 | 完成部分实验，记录不完整 | 未进行实验 |
| **理论知识** | 20% | 能推导核心公式，解释设计原理，关联学术论文 | 理解核心概念，能解释为什么 | 知道基本概念，但理解不深 | 不理解核心理论 |
| **工程能力** | 15% | 能修改代码实现新功能，处理异常情况 | 能调参优化，理解配置含义 | 能运行脚本，基本参数调整 | 无法独立运行代码 |
| **报告质量** | 15% | 报告逻辑清晰，数据完整，有深入分析和洞察 | 报告结构完整，数据充分 | 报告基本完整，缺少分析 | 无报告或内容空洞 |

### 15.2 阶段性检查点

| 周次 | 检查点 | 交付物 |
|------|--------|--------|
| 第 1 周 | 实验一、二 | 分词器评估数据 + 模型结构图 |
| 第 2 周 | 实验三、四 | 数据管线理解报告 + 预训练实验记录 |
| 第 3 周 | 实验五、六 | 优化器分析笔记 + SFT 实验记录 |
| 第 4 周 | 实验七、八 | RL 实验记录 + 部署 Demo 截图 |
| 第 5 周 | 实验九、十 | 评估结果表 + 完整实验报告初稿 |
| 第 6-8 周 | 综合实践 | 深度探索 + 报告定稿 + 面试准备 |

---

## 16. 附录：常见问题与进阶方向

### 16.1 常见问题 (FAQ)

**Q1: 没有 8×H100，能在消费级 GPU 上运行吗？**
可以。使用 `--depth=6`、`--device-batch-size=4`、`--max-seq-len=256` 等参数，单卡 3090/4090 即可运行。CPU/Mac 也可以用 `runs/runcpu.sh` 体验。

**Q2: 数据下载很慢怎么办？**
使用 `-n 1` 参数只下载 1 个分片用于实验（约 100MB）。也可以挂代理或使用 HuggingFace 镜像。

**Q3: torch.compile 报错怎么办？**
确保 PyTorch 版本 ≥ 2.0。如果是较老的 GPU（SM < 80），可能会遇到兼容性问题，可以临时去掉 `torch.compile`。

**Q4: 这个项目适合完全零基础吗？**
需要一定的 PyTorch 基础和 Transformer 理解。建议先学习 Andrej Karpathy 的 "Neural Networks: Zero to Hero" 系列视频或 "nanoGPT" 项目。

### 16.2 进阶探索方向

如果你已完成所有基础实验，可以选择以下方向深入：

| 方向 | 探索内容 | 难度 |
|------|---------|------|
| **MoE 架构** | 将 FFN 替换为 Mixture of Experts，实现稀疏激活 | ⭐⭐⭐⭐⭐ |
| **更长上下文** | 将上下文长度扩展到 8K/32K，分析训练效率变化 | ⭐⭐⭐ |
| **多模态** | 添加图像编码器，实现视觉-语言联合训练 | ⭐⭐⭐⭐⭐ |
| **推测解码** | 使用小模型预测，大模型验证，加速推理 | ⭐⭐⭐ |
| **量化部署** | 实现 GPTQ/AWQ 量化，对比推理速度和质量 | ⭐⭐⭐ |
| **多节点扩展** | 研究 FSDP/TP/PP 等多维并行策略 | ⭐⭐⭐⭐ |
| **工具使用** | 扩展工具调用（搜索、数据库、API），构建 Agent | ⭐⭐⭐ |
| **安全训练** | 添加安全约束、红队测试、价值观对齐 | ⭐⭐⭐⭐ |

### 16.3 推荐阅读

| 论文/资源 | 关联实验 | 关键概念 |
|----------|---------|---------|
| Attention Is All You Need (2017) | 实验二 | Transformer 基础 |
| RoFormer: RoPE (2021) | 实验二 | 旋转位置编码 |
| Chinchilla (2022) | 实验四 | 计算最优训练 |
| Flash Attention 3 (2024) | 实验二/八 | 高效注意力 |
| Muon Optimizer / Polar Express (2025) | 实验五 | 动量正交化 |
| DeepSeek-R1 / GRPO (2025) | 实验七 | 组相对策略优化 |
| DCLM / CORE (2024) | 实验九 | 标准化评估 |
| nanoGPT (Karpathy) | 全部 | 教学级 GPT 实现 |
| Karpathy's Zero to Hero | 全部 | 配套视频课程 |

---

> **最后的话**：
>
> 作为 HR 和指导老师，我想强调：这个项目的价值不仅在于技术本身，更在于它所体现的工程思维——**用最少的代码实现最完整的功能，用单一旋钮控制复杂系统，用理论指导工程实践**。这些能力正是顶级 AI 公司最看重的。
>
> 在面试中，不要只说你"跑通了"这个项目，而要展示你**理解为什么**：为什么选择 Muon 而不是纯 AdamW？为什么 FP8 只转换部分层？为什么 RL 在数学推理上做而不是通用对话？**"为什么"才是区分优秀工程师和调参侠的分水岭。**
>
> 祝你求职顺利！🚀

---

*本实验指导书基于 NanoChat 项目（commit dc54a1a，2026 年 6 月），项目地址：https://github.com/karpathy/nanochat*
