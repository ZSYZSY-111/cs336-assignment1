# 实验流程文档

本项目包含两个对比实验，都建立在 `prepare_data.py` 的预处理产出之上：

- **BPE 对比**：单进程 vs 并行预分词，验证并行更快
- **MHA / GQA 对比**：标准多头注意力 vs 分组查询注意力，验证 GQA 省显存、提速、loss 不掉

---

## 全流程总览

```
prepare_data.py ──────► prepared_data/ (共用)
       │
       ├──► benchmark_bpe.py ──► BPE 速度对比（独立）
       │
       ├──► train_model.py ──────► checkpoints_5m/  (MHA)
       │
       └──► train_model_gqa.py ──► checkpoints_gqa/ (GQA)
                   │
                   ▼
            compare_mha_gqa.py ──► 对比表 + 图
```

---

## 阶段 0：环境准备（只需一次）

```bash
cd /root && git clone git@github.com:ZSYZSY-111/cs336-assignment1.git
cd cs336-assignment1/related_code

pip install -e ..
pip install regex

# 验证 GPU（应输出 True + 显卡名）
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> 若 GPU 不可用：`pip install torch --index-url https://download.pytorch.org/whl/cu128`

---

## 阶段 1：数据预处理（只需一次，两实验共用）

```bash
# 确认数据在位
ls -lh /root/autodl-tmp/TinyStoriesV2-GPT4-train.txt /root/autodl-tmp/TinyStoriesV2-GPT4-valid.txt

# 训 BPE + 编码（较慢）
python prepare_data.py
```

产出：`/root/autodl-tmp/prepared_data/`
- `tokenizer.pkl`
- `train_tokens.npy`
- `val_tokens.npy`

---

## 阶段 2：BPE 对比实验（独立，可单独跑）

```bash
python benchmark_bpe.py
```

- 先跑单进程版（慢），再跑并行版，对比速度
- 关注 `Speedup`（加速倍数）和 `Results match: True`（结果一致性）
- 纯 CPU 实验，不需要 GPU

可选参数：
```bash
# 用小文件快速验证（22MB，几秒出结果）
python benchmark_bpe.py --input /root/autodl-tmp/TinyStoriesV2-GPT4-valid.txt

# 更大 vocab，对比更明显
python benchmark_bpe.py --vocab-size 10000
```

---

## 阶段 3：MHA Baseline 训练

```bash
python train_model.py
```

产出：`/root/autodl-tmp/checkpoints_5m/`（`train_log.csv` + 权重）。
看到 `peak GPU memory: xxx MB` 即完成。

---

## 阶段 4：GQA 训练

```bash
python train_model_gqa.py
```

产出：`/root/autodl-tmp/checkpoints_gqa/`（`train_log_gqa.csv` + 权重）。
与 MHA 分目录存储，不会互相覆盖。

---

## 阶段 5：MHA vs GQA 对比

```bash
python compare_mha_gqa.py --plot
```

- 读两份 CSV 日志，输出 loss / 速度 / 显存对比表
- 保存曲线图 `mha_vs_gqa_val_loss.png`

---

## 一句话顺序

```bash
python prepare_data.py              # 1. 预处理
python benchmark_bpe.py             # 2. BPE 对比
python train_model.py               # 3. MHA 训练
python train_model_gqa.py           # 4. GQA 训练
python compare_mha_gqa.py --plot    # 5. 训练对比
```

---

## 控制变量说明

### MHA vs GQA：唯一变量是 KV 头数

| 参数 | MHA | GQA | 相同？ |
|------|-----|-----|--------|
| num_heads | 8 | 8 | ✅ |
| num_kv_heads | 8（满）| 4 | ❌ **唯一变量** |
| 数据 / vocab / 层数 / lr / 步数 | 一致 | 一致 | ✅ |

> ⚠️ 两次训练须在同一台 GPU 上跑，否则显存/速度对比不公平。

### 预期结果

- **GQA**：显存 ↓、tokens/s ↑、loss 基本持平
- **并行 BPE**：speedup 显著（接近核数），结果与单进程完全一致

---

## 文件说明

| 文件 | 作用 |
|------|------|
| `prepare_data.py` | 训 BPE + 编码，生成共用数据 |
| `train_bpe_upgrade.py` | 单进程 BPE |
| `train_bpe_parallel.py` | 并行 BPE（多进程预分词） |
| `benchmark_bpe.py` | BPE 速度对比 |
| `train_transformer.py` | MHA 模型 |
| `train_transformer_gqa.py` | GQA 模型 |
| `train_model.py` | MHA 训练 |
| `train_model_gqa.py` | GQA 训练 |
| `compare_mha_gqa.py` | MHA/GQA 对比 |
