from collections import Counter
import regex as re

def text_to_byte_tokens(text):
    return [bytes([b]) for b in text.encode("utf-8")]


def split_on_special_tokens(text, special_tokens):
    if not special_tokens:
        return [text]

    pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
    return re.split(pattern, text)

GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize(text): # 预分词, "Hello, world! I'm happy." -> ["Hello", ",", " world", "!", " I", "'m", " happy", "."]
    return re.findall(GPT2_PRETOKEN_PATTERN, text)

def text_to_pretoken_byte_tokens(text): # "Hello" -> (72, 101, 108, 108, 111)
    pretokens = pretokenize(text)
    return [text_to_byte_tokens(pretoken) for pretoken in pretokens]

def get_pair_counts_from_sequences(sequences):
    counts = Counter()
    for tokens in sequences:
        for a, b in zip(tokens, tokens[1:]):
            counts[(a, b)] += 1
    return counts

def merge_pair(tokens, pair): # 按照提供的pair对token进行合并
    new_tokens = []
    i = 0

    while i < len(tokens):
        # 如果当前位置和下一个位置正好等于 pair，就合并
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            new_tokens.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1

    return new_tokens

def merge_pair_in_sequences(sequences, pair):
    return [merge_pair(tokens, pair) for tokens in sequences]

def train_bpe_pretokenized(text, num_merges): # 反复找到最多的pair, 并将它们合并，合并次数num_merges
    sequences = text_to_pretoken_byte_tokens(text)
    merges = []

    for _ in range(num_merges):
        pair_counts = get_pair_counts_from_sequences(sequences)

        if not pair_counts:
            break

        best_pair, best_count = max(pair_counts.items(), key=lambda item: (item[1], item[0]))

        sequences = merge_pair_in_sequences(sequences, best_pair)
        merges.append(best_pair)

        print(f"merge {best_pair} -> {best_pair[0] + best_pair[1]}, count={best_count}")
        print(sequences)
        print()

    return merges, sequences

def train_bpe_with_vocab_size(text, vocab_size, special_tokens): # 按照词表大小训练pbe的merge规则
    num_merges = vocab_size - 256 - len(special_tokens)

    if num_merges < 0:
        raise ValueError("vocab_size is too small")

    sequences = []

    chunks = split_on_special_tokens(text, special_tokens)

    for chunk in chunks:
        if chunk in special_tokens:
            continue

        for pretoken in pretokenize(chunk):
            sequences.append(text_to_byte_tokens(pretoken))

    merges = []

    for _ in range(num_merges):
        pair_counts = get_pair_counts_from_sequences(sequences)

        if not pair_counts:
            break

        best_pair, best_count = max(
            pair_counts.items(),
            key=lambda item: (item[1], item[0])
        )

        sequences = merge_pair_in_sequences(sequences, best_pair)
        merges.append(best_pair)

    return merges


def build_vocab_from_merges(merges, special_tokens): # 制作词表
    vocab = {}

    for i in range(256):
        vocab[i] = bytes([i])

    next_id = 256

    for special in special_tokens:
        vocab[next_id] = special.encode("utf-8")
        next_id += 1

    for pair in merges:
        vocab[next_id] = pair[0] + pair[1]
        next_id += 1

    return vocab

def apply_merges(tokens, merges):
    for pair in merges:
        tokens = merge_pair(tokens, pair)
    return tokens

def encode(text, vocab, merges, special_tokens):
    token_to_id = {token: idx for idx, token in vocab.items()}

    ids = []

    chunks = split_on_special_tokens(text, special_tokens)

    for chunk in chunks:
        if chunk == "":
            continue

        if chunk in special_tokens:
            ids.append(token_to_id[chunk.encode("utf-8")])
            continue

        for pretoken in pretokenize(chunk):
            tokens = text_to_byte_tokens(pretoken)
            tokens = apply_merges(tokens, merges)
            ids.extend(token_to_id[token] for token in tokens)

    return ids


def decode(ids, vocab):
    tokens = [vocab[i] for i in ids]
    text_bytes = b"".join(tokens)
    return text_bytes.decode("utf-8", errors="replace")

def run_train_bpe(input_path, vocab_size, special_tokens):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    merges = train_bpe_with_vocab_size(text, vocab_size, special_tokens)
    vocab = build_vocab_from_merges(merges, special_tokens)

    return vocab, merges

