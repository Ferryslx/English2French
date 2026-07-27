# 英译法神经机器翻译

基于 PyTorch 从零实现的 Transformer 与 GRU+注意力机制 的英译法对比项目。

## 概述

对比两种序列到序列架构在英法翻译任务上的表现：

- **Transformer**（Pre-LayerNorm）— 参考 "Attention Is All You Need"（Vaswani 等, 2017）
- **GRU + Bahdanau 注意力** — 基于循环神经网络的注意力基线模型

两者使用相同的数据集、损失函数和评估方式。

## 数据集

- 来源：[ManyThings.org](https://www.manythings.org/anki/) 英法平行语料库
- 预处理：SentencePiece unigram 分词（词表大小 = 6,000）
- 格式：添加 BOS/EOS 标记，填充至最大长度 20
- 规模：63,594 个句对（58,474 训练 / 5,120 验证）

预处理脚本：`src/data_preprocess.py`

## 模型架构

### Transformer（`component/transformer_complete.py`）

| 超参数 | 取值 |
|---|---|
| d_model | 256 |
| d_ff | 1024 |
| 注意力头数 | 4 |
| 层数 | 4 |
| Dropout | 0.2 |
| 归一化方式 | Pre-LN |
| 优化器 | Adam (β₁=0.9, β₂=0.98, ε=1e-9) |
| 学习率策略 | Noam warmup（4,000 步） |
| 梯度裁剪 | 5.0 |
| 参数量 | ~12.0M |

### GRU + 注意力（`src/train_comparison.py`）

| 超参数 | 取值 |
|---|---|
| 词嵌入维度 | 256 |
| 隐藏层维度 | 256 |
| GRU 层数 | 1 |
| Dropout | 0.2 |
| 注意力机制 | Bahdanau（加性注意力） |
| 优化器 | Adam (lr=0.001) |
| 参数量 | ~10.3M |

### 训练参数（共用）

| 超参数 | 取值 |
|---|---|
| 最大序列长度 | 20 |
| 批大小 | 64 |
| 训练轮数 | 20 |
| 损失函数 | NLLLoss（忽略 PAD） |
| 运行设备 | CPU / CUDA |

## 项目结构

```
English2French/
├── component/
│   └── transformer_complete.py   # Transformer 完整实现
├── src/
│   ├── data_preprocess.py         # SentencePiece 数据预处理
│   ├── train_comparison.py        # Transformer vs GRU 训练
│   ├── ablation_transformer.py    # 消融实验（exp1/2/3/5）
│   └── ablation_exp4_ln.py        # Pre-LN vs Post-LN 对比
├── utils/
│   └── log.py                     # 日志工具
├── data/
│   ├── eng-fra-v2.txt             # 原始平行语料
│   ├── train.en.sp / train.fr.sp  # 分词后数据
│   └── spm.model                  # SentencePiece 模型
├── model/                         # 训练好的模型权重 (.pth)
├── loss/                          # 训练/验证损失 CSV
├── log/                           # 训练日志
└── figures/                       # 损失曲线图
```

## 实验结果（第 20 轮）

### Transformer vs GRU

| 模型 | 训练损失 | 验证损失 | 参数量 |
|---|---|---|---|
| **Transformer**（Pre-LN） | 0.1773 | **0.1515** | 11,987,824 |
| **GRU + 注意力** | 0.1647 | 0.1671 | ~10.3M |

Transformer 验证损失更低，泛化能力更强。GRU 出现轻微过拟合（训练损失低于验证损失）。

### 消融实验

| 实验 | 变体 | 训练损失 | 验证损失 |
|---|---|---|---|
| 基线 | d_model=256, N=4, warmup, dropout=0.2 | 0.1773 | 0.1515 |
| exp1_no_warmup | 恒定 lr=0.0003，无 warmup | 0.2415 | 0.2062 |
| exp2_n1 | 单层 N=1 | 0.2769 | 0.2108 |
| exp3_d128 | d_model=128, d_ff=512 | 0.3323 | 0.2515 |
| exp5_dropout0 | dropout=0.0 | 0.1329 | **0.1388** |

主要发现：
- 移除 Noam warmup 后收敛明显变差——warmup 对 Transformer 训练稳定性至关重要
- 单层 Transformer 欠拟合——即使在此规模下深度仍然重要
- 减小模型宽度（d_model=128）导致容量瓶颈
- 零 dropout 取得最低损失，说明当前参数量下过拟合不是主要问题

### Pre-LN vs Post-LN（实验 4）

| 变体 | 训练损失 | 验证损失 |
|---|---|---|
| **Pre-LN**（默认） | 0.1769 | **0.1582** |
| **Post-LN**（原版 Vaswani） | 0.2680 | 0.1824 |

Pre-LN 全面优于 Post-LN，验证了 Pre-LayerNorm 在小规模 Transformer 中的训练稳定性优势。

## 使用说明

### 数据预处理

```bash
python src/data_preprocess.py
```

### 训练 Transformer 和 GRU

```bash
python src/train_comparison.py
```

### 运行消融实验

```bash
python src/ablation_transformer.py   # 实验 1、2、3、5
python src/ablation_exp4_ln.py       # 实验 4：Pre-LN vs Post-LN
```

日志输出到 `log/`，损失数据输出到 `loss/`，模型权重输出到 `model/`。

## 依赖

- Python 3.8+
- PyTorch
- sentencepiece
- tqdm
