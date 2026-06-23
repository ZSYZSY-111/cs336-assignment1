# BPE 三个版本对比

本项目中 BPE（Byte Pair Encoding）tokenizer 经历了三次迭代：

| 版本 | 文件 | 定位 |
|------|------|------|
| V1 | `train_bpe.py` | 原始朴素实现，正确但慢 |
| V2 | `train_bpe_upgrade.py` | 算法层优化，单进程大幅加速 |
| V3 | `train_bpe_parallel.py` | 系统层优化，多进程并行 |

---

## BPE 算法简述

BPE 分两个阶段：

1. **预分词（pretokenization）**：读文本 → 按特殊 token 切分 → 正则预分词 → 统计每个 pretoken 的出现次数
2. **Merge 循环**：重复 N 次「找最频繁的相邻 token 对 → 合并成新 token」

两次升级分别针对这两个阶段的瓶颈。

---

## V1 → V2：算法层优化（`train_bpe.py` → `train_bpe_upgrade.py`）

### 升级点 1：预分词结果去重 + 词频统计

**V1（慢）**：每个 pretoken 出现一次就存一个独立序列，内存随文本大小线性增长。

```python
# V1：有多少 pretoken 就存多少序列（大量重复）
for pretoken in pretokenize(chunk):
    sequences.append(text_to_byte_tokens(pretoken))
# sequences 可能有几百万个，大量重复
```

**V2（快）**：用 `Counter` 去重，只存**唯一的** pretoken，记录出现次数。

```python
# V2：去重后只存不同的 pretoken + 出现次数
pretoken_counts: Counter[tuple[int, ...]] = Counter()
for pretoken in pretokenize(chunk):
    pretoken_counts[text_to_byte_ids(pretoken)] += 1
# 可能只有几万个唯一 pretoken，内存降几个数量级
```

**效果**：内存从 O(文本 token 总数) 降到 O(唯一 pretoken 数)，对大语料极为关键。

---

### 升级点 2：Merge 循环从 O(N) 全扫变成 O(affected) 增量更新

**V1（慢）**：每次 merge 后**重新扫描全部序列**，计算所有 pair 的频次。

```python
# V1：每轮 merge 都全量重算，O(N × M) 总复杂度
for _ in range(num_merges):
    pair_counts = get_pair_counts_from_sequences(sequences)   # 全量扫描
    best_pair = max(pair_counts.items(), key=...)
    sequences = merge_pair_in_sequences(sequences, best_pair) # 全量 rebuild
```

**V2（快）**：维护 `pair_to_words` 映射，每次 merge 只更新**受影响的词**的 pair 计数。

```python
# V2：只更新受影响的词
affected_words = list(pair_to_words.get(best_pair, set()))
for word_idx in affected_words:
    # 减去旧 pair 计数，加上新 pair 计数
    ...
```

---

### 升级点 3：最优 pair 选择从 O(P) 线性扫变成 O(log P) 堆

**V1（慢）**：每轮用 `max()` 遍历所有 pair 找最大值，O(P) 时间。

```python
# V1
best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
```

**V2（快）**：用最大堆（`heapq`）维护 pair 优先级，弹出最大值 O(log P)。

```python
# V2：堆顶就是最优 pair，弹出 O(log P)
while heap:
    candidate = heappop(heap)
    # 验证 candidate 是否仍有效（懒删除策略）
    if current_count == candidate.count ...:
        best_pair = candidate.pair
        break
```

---

### 升级点 4：encode 中的 apply_merges

**V1（慢）**：对每个 pretoken，把 merges 列表**从头到尾逐个试**，O(M) 每 pretoken。

```python
# V1：线性遍历所有 merge 规则
def apply_merges(tokens, merges):
    for pair in merges:
        tokens = merge_pair(tokens, pair)
    return tokens
```

**V2（快）**：用 `merge_ranks` 字典，每轮只找当前 token 对中**rank 最小**的 merge。

```python
# V2：只处理当前存在的 pair，不盲目遍历
merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
while True:
    ranked_pairs = [(merge_ranks[pair], pair) for pair in pairs if pair in merge_ranks]
    if not ranked_pairs:
        return tokens
    _, best_pair = min(ranked_pairs)
    tokens = merge_pair(tokens, best_pair)
```

---

### 升级点 5：special_tokens 切分更健壮

**V1**：特殊 token 模式不排序，长 token 可能被短 token 的前缀误匹配。

**V2**：按长度降序排列，确保长 token 优先匹配。

```python
# V2：sorted by length descending
special_tokens = sorted(special_tokens, key=len, reverse=True)
```

---

## V2 → V3：系统层优化（`train_bpe_upgrade.py` → `train_bpe_parallel.py`）

### 升级点 6：预分词并行化（最大的瓶颈）

V2 的预分词阶段是单进程读整个文件，在大语料（如 OWT 12GB）上是最大瓶颈。

**V2（单进程）**：

```python
# 一次性读全部文件，单核处理
with open(input_path, encoding="utf-8") as f:
    text = f.read()   # 读 12GB 进内存
for chunk in split_on_special_tokens(text, special_tokens):
    for pretoken in pretokenize(chunk):
        pretoken_counts[...] += 1
```

**V3（多进程）**：

```python
# 文件切成 N 块，N 个进程并行跑
boundaries = find_chunk_boundaries(f, n, split_token)  # 按特殊 token 边界切分
with Pool(processes=n) as pool:
    results = pool.map(_count_chunk, chunks)            # 并行预分词
pretoken_counts = sum(results, Counter())               # 合并计数
```

**关键设计**：
- 以 `<|endoftext|>` 为边界切分，保证不在文档中间切断（结果正确）
- 各进程只读自己那块字节（内存从 O(全文) 降到 O(单块)）
- `Counter` 相加可交换，不需要考虑顺序（结果与单进程完全一致）
- Merge 阶段**完全不变**，只有预分词并行了

### 升级点 7：内存占用大幅降低

| 版本 | 内存模式 |
|------|---------|
| V1 | 全文读入 + 全量展开序列（最大）|
| V2 | 全文读入 + 去重后的词频表（中等）|
| V3 | 每进程只读一块（最小）|

---

## 三版本综合对比

| 维度 | V1 | V2 | V3 |
|------|----|----|-----|
| 预分词去重 | ❌ 全量展开 | ✅ Counter 去重 | ✅ Counter 去重 |
| Merge 更新策略 | ❌ 全量重算 | ✅ 增量更新 | ✅ 增量更新（同 V2）|
| 最优 pair 选择 | ❌ O(P) max 扫描 | ✅ O(log P) 堆 | ✅ O(log P) 堆（同 V2）|
| encode apply_merges | ❌ 线性遍历 | ✅ rank 字典 | ✅ rank 字典（同 V2）|
| 文件读取 | ❌ 全量读入内存 | ❌ 全量读入内存 | ✅ 分块读取 |
| CPU 利用 | 单核 | 单核 | ✅ 多核并行 |
| 正确性 | ✅ | ✅ | ✅ 与 V2 结果完全一致 |

---

## 每次升级解决的核心问题

```
V1（朴素实现）
│  问题：
│  ① 大量重复序列浪费内存（未去重）
│  ② 每轮 merge 全量重算 pair 计数
│  ③ 最优 pair 线性扫描
│
▼
V2（算法优化）
│  解决：① ② ③  → 单进程下已是高效实现
│  新瓶颈：预分词仍是单进程，12GB 文本处理很慢
│
▼
V3（并行优化）
   解决：预分词并行化 → 充分利用多核 CPU
   Merge 阶段不变（已是 V2 的高效算法）
```

---

## 当前使用情况

| 脚本 | 使用版本 |
|------|---------|
| `prepare_data.py` | V3（并行 BPE 训练）+ V2（encode）|
| `benchmark_bpe.py` | V2 vs V3 速度对比 |
| `train_model.py` / `train_model_gqa.py` | 加载 `prepared_data/`，不直接调 BPE |
