"""Parallel BPE trainer - drop-in replacement for train_bpe_upgrade.py.

Pretokenization is parallelized across CPU cores; the merge phase is identical.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from multiprocessing import Pool
from os import PathLike

import regex as re

from train_bpe_upgrade import (
    GPT2_PRETOKEN_PATTERN,
    PairPriority,
    _build_vocab,
    _pair_bytes,
    decode,
    encode,
    pretokenize,
    split_on_special_tokens,
    text_to_byte_ids,
)


def find_chunk_boundaries(
    file, num_chunks: int, split_special: bytes = b"<|endoftext|>"
) -> list[int]:
    """Find byte offsets that split the file into num_chunks roughly equal parts,
    aligned to the nearest split_special token boundary."""
    file.seek(0, 2)
    file_size = file.tell()
    chunk_size = file_size // num_chunks

    boundaries = [0]
    for i in range(1, num_chunks):
        pos = i * chunk_size
        file.seek(pos)
        # Advance to the next split_special boundary
        remaining = file.read(10 * len(split_special) + 1024)
        idx = remaining.find(split_special)
        if idx != -1:
            boundaries.append(pos + idx + len(split_special))
        else:
            boundaries.append(pos)
    boundaries.append(file_size)
    return boundaries


def _count_chunk(args: tuple) -> Counter:
    """Worker: count pretoken frequencies in one file chunk."""
    input_path, start, end, special_tokens = args
    special_token_set = set(special_tokens)
    counts: Counter[tuple[int, ...]] = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)

    text = raw.decode("utf-8", errors="ignore")
    for chunk in split_on_special_tokens(text, special_tokens):
        if chunk == "" or chunk in special_token_set:
            continue
        for pretoken in pretokenize(chunk):
            counts[text_to_byte_ids(pretoken)] += 1
    return counts


def train_bpe_parallel(
    input_path: str | PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = special_tokens or []
    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        raise ValueError("vocab_size is too small")

    n = num_processes or os.cpu_count() or 1

    # Find chunk boundaries aligned to <|endoftext|>
    split_token = b"<|endoftext|>" if "<|endoftext|>" in special_tokens else b"\n\n"
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, n, split_token)

    chunks = [
        (input_path, boundaries[i], boundaries[i + 1], special_tokens)
        for i in range(len(boundaries) - 1)
        if boundaries[i] < boundaries[i + 1]
    ]

    # Parallel pretokenization
    with Pool(processes=n) as pool:
        results = pool.map(_count_chunk, chunks)

    pretoken_counts: Counter[tuple[int, ...]] = Counter()
    for c in results:
        pretoken_counts += c

    # ---- merge phase (identical to train_bpe_upgrade) ----
    words = [list(pretoken) for pretoken in pretoken_counts]
    word_freqs = list(pretoken_counts.values())
    token_bytes = [bytes([i]) for i in range(256)]

    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_to_words: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    heap: list[PairPriority] = []

    def add_pair(pair, word_idx, amount):
        pair_counts[pair] += amount
        pair_to_words[pair].add(word_idx)

    def remove_pair(pair, word_idx, amount):
        new_count = pair_counts[pair] - amount
        if new_count <= 0:
            pair_counts.pop(pair, None)
        else:
            pair_counts[pair] = new_count
        words_with_pair = pair_to_words.get(pair)
        if words_with_pair is not None:
            words_with_pair.discard(word_idx)
            if not words_with_pair:
                pair_to_words.pop(pair, None)

    def push_pair(pair):
        count = pair_counts.get(pair, 0)
        if count > 0:
            heappush(heap, PairPriority(count, _pair_bytes(pair, token_bytes), pair))

    for word_idx, word in enumerate(words):
        freq = word_freqs[word_idx]
        seen_pairs = set()
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freq
            seen_pairs.add(pair)
        for pair in seen_pairs:
            pair_to_words[pair].add(word_idx)

    for pair, count in pair_counts.items():
        heappush(heap, PairPriority(count, _pair_bytes(pair, token_bytes), pair))

    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        best_pair = None
        best_count = 0
        while heap:
            candidate = heappop(heap)
            current_count = pair_counts.get(candidate.pair, 0)
            current_bytes = _pair_bytes(candidate.pair, token_bytes)
            if current_count == candidate.count and current_bytes == candidate.pair_bytes:
                best_pair = candidate.pair
                best_count = current_count
                break
        if best_pair is None or best_count == 0:
            break

        best_pair_bytes = _pair_bytes(best_pair, token_bytes)
        merged_token_id = len(token_bytes)
        token_bytes.append(best_pair_bytes[0] + best_pair_bytes[1])
        merges.append(best_pair_bytes)

        affected_words = list(pair_to_words.get(best_pair, set()))
        changed_pairs: set[tuple[int, int]] = set()
        for word_idx in affected_words:
            word = words[word_idx]
            freq = word_freqs[word_idx]
            if not any(pair == best_pair for pair in zip(word, word[1:])):
                continue
            old_seen_pairs = set(zip(word, word[1:]))
            for pair in old_seen_pairs:
                remove_pair(pair, word_idx, freq * sum(1 for p in zip(word, word[1:]) if p == pair))
                changed_pairs.add(pair)
            merged_word: list[int] = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == best_pair:
                    merged_word.append(merged_token_id)
                    i += 2
                else:
                    merged_word.append(word[i])
                    i += 1
            words[word_idx] = merged_word
            new_pair_counts = Counter(zip(merged_word, merged_word[1:]))
            for pair, pair_count in new_pair_counts.items():
                add_pair(pair, word_idx, freq * pair_count)
                changed_pairs.add(pair)
        for pair in changed_pairs:
            push_pair(pair)

    return _build_vocab(token_bytes, merges, special_tokens), merges


def run_train_bpe(
    input_path: str | PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return train_bpe_parallel(input_path, vocab_size, special_tokens)
