# CS336 Assignment 1: From BPE Tokenizer to Tiny Transformer LM

这个仓库是 Stanford CS336 Assignment 1 的实验实现。项目从字节级 BPE tokenizer 开始，完成数据预处理、decoder-only Transformer 语言模型训练、loss/吞吐/显存日志记录，并额外加入两个对比实验：

- **BPE 实现演进**：朴素 BPE -> 单进程增量优化 -> 多进程并行预分词。
- **MHA vs GQA**：标准 Multi-Head Attention 与 Grouped-Query Attention 的训练效率、显存和 loss 对比。

主要实验代码在 `related_code/` 目录中；仓库根目录保留了 CS336 作业框架、测试和 handout。

## 实验流程总览

```text
raw text
  |
  v
related_code/prepare_data.py
  |-- train_bpe_parallel.py  训练 BPE tokenizer
  |-- train_bpe_upgrade.py   编码 train/val 文本
  v
prepared_data/
  |-- tokenizer.pkl
  |-- train_tokens.npy
  `-- val_tokens.npy
  |
  |--------------------------------------------|
  v                                            v
related_code/train_model.py              related_code/train_model_gqa.py
MHA baseline                              GQA variant
  |                                            |
  v                                            v
checkpoints_5m/                            checkpoints_gqa/
train_log.csv                              train_log_gqa.csv
  |                                            |
  |---------------------|----------------------|
                        v
          related_code/compare_mha_gqa.py
          loss / speed / memory 对比
```

完整实验通常按下面顺序运行：

```bash
cd related_code
python prepare_data.py
python benchmark_bpe.py
python train_model.py
python train_model_gqa.py
python compare_mha_gqa.py --plot
```

## 环境准备

建议使用 Python 3.10+。主要依赖是 `torch`、`numpy`、`regex`，画图和对比曲线需要 `matplotlib`。

```bash
pip install torch numpy regex matplotlib
```

如果要运行作业自带测试，可以在仓库根目录安装本项目：

```bash
pip install -e .
pytest
```

原始作业推荐使用 `uv`：

```bash
uv run pytest
```

训练脚本默认面向 GPU 环境，`related_code/train_model.py` 和 `related_code/train_model_gqa.py` 的 `CONFIG["device"]` 默认为 `"cuda"`。如果只在本地 CPU/MPS 上调试，需要把配置里的 `device`、数据路径、训练步数和 batch size 改小。

## 数据准备

CS336 Assignment 1 使用 TinyStories 和 OpenWebText sample。可以把数据放在仓库根目录的 `data/` 下：

```text
data/
  TinyStoriesV2-GPT4-train.txt
  TinyStoriesV2-GPT4-valid.txt
  owt_train.txt
  owt_valid.txt
```

`related_code/prepare_data.py` 默认优先使用 AutoDL 风格路径：

```text
/root/autodl-tmp/TinyStoriesV2-GPT4-train.txt
/root/autodl-tmp/TinyStoriesV2-GPT4-valid.txt
/root/autodl-tmp/prepared_data/
```

如果默认输入路径不存在，脚本会用同名文件回退到 `related_code/data/`。也可以显式指定输入输出：

```bash
cd related_code
python prepare_data.py \
  --train-text data/TinyStoriesV2-GPT4-train.txt \
  --val-text data/TinyStoriesV2-GPT4-valid.txt \
  --out-dir prepared_data \
  --vocab-size 512
```

该步骤会完成两件事：

1. 使用并行 BPE 训练 tokenizer，保存 `tokenizer.pkl`。
2. 将训练集和验证集编码成 token id，保存为 `train_tokens.npy` 和 `val_tokens.npy`。

后续模型训练直接读取 `.npy`，避免每次训练前重复跑 BPE 和编码。

## BPE Tokenizer 实验

项目里保留了三版 BPE：

| 版本 | 文件 | 作用 |
| --- | --- | --- |
| V1 | `related_code/train_bpe.py` | 朴素实现，逻辑直观但慢 |
| V2 | `related_code/train_bpe_upgrade.py` | 使用词频去重、堆和增量更新优化 merge |
| V3 | `related_code/train_bpe_parallel.py` | 在 V2 基础上并行化预分词和文件分块读取 |

BPE 的核心流程是：

1. 按 special token 切分文本。
2. 使用 GPT-2 风格正则做 pretokenization。
3. 将 pretoken 转成 UTF-8 byte 序列。
4. 反复统计最常见的相邻 byte/token pair，并把它合并成新 token。
5. 根据 256 个原始 byte、special tokens 和 merge 结果构造词表。

并行版本只并行预分词计数阶段；merge 阶段保持与优化单进程版本一致，因此可以验证结果一致性。

运行 BPE benchmark：

```bash
cd related_code
python benchmark_bpe.py
```

可选地指定数据和词表大小：

```bash
python benchmark_bpe.py --input data/TinyStoriesV2-GPT4-valid.txt --vocab-size 10000
```

预期关注点：

- `Results match  : merges=True, vocab=True`：并行版与单进程优化版结果一致。
- `Speedup`：并行预分词带来的加速倍数。

更详细的 BPE 版本解释见 `related_code/BPE_VERSIONS.md`。

## Transformer 训练

`related_code/train_transformer.py` 实现了一个 decoder-only TransformerLM，主要组件包括：

- `Embedding` 和 `Linear`：手写参数初始化。
- `RMSNorm`：归一化层。
- `ROPE`：旋转位置编码。
- `MultiHeadAttention`：带 causal mask 的多头自注意力。
- `SwiGLU`：前馈网络。
- `TransformerBlock`：pre-norm attention + FFN 残差结构。
- `TransformerLM`：token embedding、多个 block、final norm 和 lm head。

训练入口是 `related_code/train_model.py`。默认配置：

| 参数 | 默认值 |
| --- | --- |
| vocab size | 512 |
| context length | 256 |
| batch size | 32 |
| layers | 6 |
| d_model | 512 |
| heads | 8 |
| d_ff | 2048 |
| optimizer | 手写 AdamW |
| lr schedule | warmup + cosine decay |
| train steps | 20000 |

运行：

```bash
cd related_code
python train_model.py
```

训练过程中会：

1. 从 `prepared_data_dir` 加载 `tokenizer.pkl`、`train_tokens.npy`、`val_tokens.npy`。
2. 随机采样长度为 `context_length` 的 batch。
3. 前向计算 next-token logits。
4. 使用手写 cross entropy 计算 loss。
5. 反向传播，做 gradient clipping。
6. 用手写 AdamW 更新参数。
7. 周期性评估 train/val loss。
8. 写入 CSV 日志并保存 checkpoint。

默认输出：

```text
/root/autodl-tmp/checkpoints_5m/
  train_log.csv
  tiny_step_1000.pt
  tiny_step_2000.pt
  ...
  tiny_final.pt
```

## GQA 对比实验

`related_code/train_transformer_gqa.py` 和 `related_code/train_model_gqa.py` 实现 Grouped-Query Attention 版本。核心变量是：

| 参数 | MHA | GQA |
| --- | --- | --- |
| `num_heads` | 8 | 8 |
| `num_kv_heads` | 8 | 4 |
| 其他训练配置 | 相同 | 相同 |

GQA 的思想是让多组 query heads 共享较少的 key/value heads，从而降低 KV 投影和注意力中的显存/计算开销。为了公平比较，两次训练共用同一份 `prepared_data/`，并保持层数、hidden size、学习率、batch size、训练步数等配置一致。

运行 GQA：

```bash
cd related_code
python train_model_gqa.py
```

默认输出：

```text
/root/autodl-tmp/checkpoints_gqa/
  train_log_gqa.csv
  tiny_step_1000.pt
  tiny_step_2000.pt
  ...
  tiny_final.pt
```

完成 MHA 和 GQA 两次训练后，对比日志：

```bash
cd related_code
python compare_mha_gqa.py --plot
```

该脚本会读取两份 CSV，输出：

- final train loss
- final validation loss
- average ms/step
- average tokens/sec
- peak GPU memory

加上 `--plot` 后会保存验证 loss 曲线图 `mha_vs_gqa_val_loss.png`。预期现象是 GQA 显存更低、吞吐更高，同时 loss 与 MHA 接近。

## 文本生成

训练完成后，可以用 `related_code/generate_text.py` 从 checkpoint 采样：

```bash
cd related_code
python generate_text.py \
  --checkpoint /root/autodl-tmp/checkpoints_5m/tiny_final.pt \
  --tokenizer-path /root/autodl-tmp/prepared_data/tokenizer.pkl \
  --prompt "Once upon a time" \
  --max-new-tokens 100 \
  --temperature 0.8 \
  --top-k 50
```

脚本会从 checkpoint 自动推断 `vocab_size`、`d_model`、`d_ff` 和层数。注意 `--num-heads`、`--context-length`、`--rope-theta` 需要与训练配置匹配。

## Loss 可视化

单次训练日志可以用 `related_code/plot_loss.py` 画图：

```bash
cd related_code
python plot_loss.py /root/autodl-tmp/checkpoints_5m/train_log.csv
```

也可以只画前若干步：

```bash
python plot_loss.py /root/autodl-tmp/checkpoints_5m/train_log.csv --max-step 5000
```

输出默认为同目录下的 `<csv_stem>_loss.png`。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `related_code/prepare_data.py` | 训练 BPE tokenizer，并把 train/val 文本编码成 `.npy` |
| `related_code/train_bpe.py` | 朴素 BPE 实现 |
| `related_code/train_bpe_upgrade.py` | 优化版 BPE，实现高效 merge 和 encode/decode |
| `related_code/train_bpe_parallel.py` | 多进程并行预分词 BPE |
| `related_code/benchmark_bpe.py` | 比较单进程优化 BPE 与并行 BPE 的速度和一致性 |
| `related_code/data.py` | 随机采样 next-token prediction batch |
| `related_code/nn_util.py` | softmax、cross entropy、gradient clipping |
| `related_code/optimizer.py` | 手写 AdamW 和 cosine learning-rate schedule |
| `related_code/train_transformer.py` | MHA TransformerLM |
| `related_code/train_transformer_gqa.py` | GQA TransformerLM |
| `related_code/train_model.py` | MHA baseline 训练脚本 |
| `related_code/train_model_gqa.py` | GQA 训练脚本 |
| `related_code/serialization.py` | checkpoint 保存/加载工具 |
| `related_code/generate_text.py` | 从训练好的 checkpoint 进行文本生成 |
| `related_code/plot_loss.py` | 从 CSV 日志绘制 loss 曲线 |
| `related_code/compare_mha_gqa.py` | 对比 MHA/GQA 的 loss、速度和显存 |
| `related_code/EXPERIMENTS.md` | 更偏运行顺序的实验记录 |
| `related_code/BPE_VERSIONS.md` | BPE 三个版本的实现细节说明 |

## 主要产物

| 产物 | 来源 | 用途 |
| --- | --- | --- |
| `prepared_data/tokenizer.pkl` | `prepare_data.py` | tokenizer 的 vocab、merges、special tokens |
| `prepared_data/train_tokens.npy` | `prepare_data.py` | 训练集 token ids |
| `prepared_data/val_tokens.npy` | `prepare_data.py` | 验证集 token ids |
| `checkpoints_5m/train_log.csv` | `train_model.py` | MHA loss、速度、显存日志 |
| `checkpoints_gqa/train_log_gqa.csv` | `train_model_gqa.py` | GQA loss、速度、显存日志 |
| `tiny_step_*.pt` / `tiny_final.pt` | 训练脚本 | 模型和优化器 checkpoint |
| `*_loss.png` | `plot_loss.py` / `compare_mha_gqa.py` | loss 曲线 |

## 复现实验时的注意事项

- `train_model.py` 和 `train_model_gqa.py` 中的路径默认是 `/root/autodl-tmp/...`，本地运行时请改成自己的路径或先用命令行参数生成 `prepared_data/`。
- MHA/GQA 对比应在同一台机器、同一张 GPU、相同 batch size 下运行，否则速度和显存对比不公平。
- `generate_text.py` 使用的 tokenizer 必须来自同一次训练或至少拥有相同 vocab size 和 merge 规则。
- 如果只是检查代码正确性，可以先把 `num_iters`、`batch_size`、`d_model`、`num_layers` 调小做 smoke test。
- `tests/` 是作业原始测试，可用于验证基础组件实现。

