from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from os import PathLike

import regex as re


GPT2_PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def split_on_special_tokens(text: str, special_tokens: list[str] | None) -> list[str]:
    if not special_tokens:
        return [text]

    special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
    return re.split(pattern, text)


def pretokenize(text: str) -> list[str]:
    return re.findall(GPT2_PRETOKEN_PATTERN, text)


def text_to_byte_ids(text: str) -> tuple[int, ...]:
    return tuple(text.encode("utf-8"))


@dataclass(frozen=True)
class PairPriority:
    count: int
    pair_bytes: tuple[bytes, bytes]
    pair: tuple[int, int]

    def __lt__(self, other: "PairPriority") -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.pair_bytes > other.pair_bytes


def _pair_bytes(pair: tuple[int, int], token_bytes: list[bytes]) -> tuple[bytes, bytes]:
    return token_bytes[pair[0]], token_bytes[pair[1]]


def _build_vocab(
    token_bytes: list[bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256

    for special_token in special_tokens:
        vocab[next_id] = special_token.encode("utf-8")
        next_id += 1

    for token_1, token_2 in merges:
        vocab[next_id] = token_1 + token_2
        next_id += 1

    return vocab


def train_bpe(
    input_path: str | PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens = special_tokens or []
    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        raise ValueError("vocab_size is too small")

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    special_token_set = set(special_tokens)
    pretoken_counts: Counter[tuple[int, ...]] = Counter()
    for chunk in split_on_special_tokens(text, special_tokens):
        if chunk == "" or chunk in special_token_set:
            continue
        for pretoken in pretokenize(chunk):
            pretoken_counts[text_to_byte_ids(pretoken)] += 1

    words = [list(pretoken) for pretoken in pretoken_counts]
    word_freqs = list(pretoken_counts.values())
    token_bytes = [bytes([token_id]) for token_id in range(256)]

    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_to_words: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    heap: list[PairPriority] = []

    def add_pair(pair: tuple[int, int], word_idx: int, amount: int) -> None:
        pair_counts[pair] += amount
        pair_to_words[pair].add(word_idx)

    def remove_pair(pair: tuple[int, int], word_idx: int, amount: int) -> None:
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

    def push_pair(pair: tuple[int, int]) -> None:
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
        best_pair: tuple[int, int] | None = None
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
    return train_bpe(input_path, vocab_size, special_tokens)


def merge_pair(tokens: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged_tokens: list[bytes] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            merged_tokens.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            merged_tokens.append(tokens[i])
            i += 1
    return tuple(merged_tokens)


def apply_merges(tokens: tuple[bytes, ...], merges: list[tuple[bytes, bytes]]) -> tuple[bytes, ...]:
    merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

    while True:
        pairs = list(zip(tokens, tokens[1:]))
        ranked_pairs = [(merge_ranks[pair], pair) for pair in pairs if pair in merge_ranks]
        if not ranked_pairs:
            return tokens

        _, best_pair = min(ranked_pairs)
        tokens = merge_pair(tokens, best_pair)


def encode(
    text: str,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> list[int]:
    special_tokens = special_tokens or []
    special_token_set = set(special_tokens)
    token_to_id = {token: token_id for token_id, token in vocab.items()}
    ids: list[int] = []

    for chunk in split_on_special_tokens(text, special_tokens):
        if chunk == "":
            continue
        if chunk in special_token_set:
            ids.append(token_to_id[chunk.encode("utf-8")])
            continue

        for pretoken in pretokenize(chunk):
            tokens = tuple(bytes([byte]) for byte in pretoken.encode("utf-8"))
            ids.extend(token_to_id[token] for token in apply_merges(tokens, merges))

    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    return b"".join(vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")
