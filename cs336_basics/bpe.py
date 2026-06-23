from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
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


def text_to_byte_tokens(text: str) -> tuple[bytes, ...]:
    return tuple(bytes([byte]) for byte in text.encode("utf-8"))


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
    for pair in merges:
        tokens = merge_pair(tokens, pair)
    return tokens


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
    word_counts: Counter[tuple[bytes, ...]] = Counter()
    for chunk in split_on_special_tokens(text, special_tokens):
        if chunk == "" or chunk in special_token_set:
            continue
        for pretoken in pretokenize(chunk):
            word_counts[text_to_byte_tokens(pretoken)] += 1

    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        for tokens, count in word_counts.items():
            for pair in zip(tokens, tokens[1:]):
                pair_counts[pair] += count

        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        word_counts = Counter(
            {
                merge_pair(tokens, best_pair): count
                for tokens, count in word_counts.items()
            }
        )
        merges.append(best_pair)

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for special_token in special_tokens:
        vocab[next_id] = special_token.encode("utf-8")
        next_id += 1
    for token_1, token_2 in merges:
        vocab[next_id] = token_1 + token_2
        next_id += 1

    return vocab, merges


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_token_set = set(self.special_tokens)
        self.token_to_id = {token: token_id for token_id, token in vocab.items()}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        for chunk in split_on_special_tokens(text, self.special_tokens):
            if chunk == "":
                continue
            if chunk in self.special_token_set:
                ids.append(self.token_to_id[chunk.encode("utf-8")])
                continue
            for pretoken in pretokenize(chunk):
                tokens = apply_merges(text_to_byte_tokens(pretoken), self.merges)
                ids.extend(self.token_to_id[token] for token in tokens)

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: Iterable[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")
