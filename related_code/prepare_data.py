from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from train_bpe_upgrade import encode
from train_bpe_parallel import run_train_bpe


ROOT = Path(__file__).resolve().parent


DEFAULT_TRAIN_TEXT_NAME = "TinyStoriesV2-GPT4-train.txt"
DEFAULT_VAL_TEXT_NAME = "TinyStoriesV2-GPT4-valid.txt"
DEFAULT_TRAIN_TEXT = Path("/root/autodl-tmp") / DEFAULT_TRAIN_TEXT_NAME
DEFAULT_VAL_TEXT = Path("/root/autodl-tmp") / DEFAULT_VAL_TEXT_NAME
DEFAULT_OUT_DIR = Path("/root/autodl-tmp/prepared_data")
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]


def save_tokenizer(
    out_dir: Path,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
) -> None:
    tokenizer_path = out_dir / "tokenizer.pkl"
    with tokenizer_path.open("wb") as f:
        pickle.dump(
            {
                "vocab": vocab,
                "merges": merges,
                "special_tokens": special_tokens,
            },
            f,
        )


def encode_to_npy(
    text_path: Path,
    out_path: Path,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
) -> None:
    print(f"encoding {text_path}...", flush=True)
    text = text_path.read_text(encoding="utf-8")
    token_ids = encode(text, vocab, merges, special_tokens)
    tokens = np.array(token_ids, dtype=np.int64)
    np.save(out_path, tokens)
    print(f"saved {len(tokens)} tokens to {out_path}", flush=True)


def prepare_data(
    train_text_path: Path,
    val_text_path: Path,
    out_dir: Path,
    vocab_size: int,
    special_tokens: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"training BPE on {train_text_path} with vocab_size={vocab_size}...", flush=True)
    vocab, merges = run_train_bpe(train_text_path, vocab_size, special_tokens)
    save_tokenizer(out_dir, vocab, merges, special_tokens)
    print(f"saved tokenizer to {out_dir / 'tokenizer.pkl'}", flush=True)

    encode_to_npy(train_text_path, out_dir / "train_tokens.npy", vocab, merges, special_tokens)
    encode_to_npy(val_text_path, out_dir / "val_tokens.npy", vocab, merges, special_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BPE and encode train/val text into cached token files.")
    parser.add_argument("--train-text", type=Path, default=DEFAULT_TRAIN_TEXT)
    parser.add_argument("--val-text", type=Path, default=DEFAULT_VAL_TEXT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--special-token", action="append", dest="special_tokens", default=None)
    return parser.parse_args()


def resolve_text_path(path: Path) -> Path:
    if path.exists():
        return path

    data_path = ROOT / "data" / path.name
    if data_path.exists():
        return data_path

    return path


def main() -> None:
    args = parse_args()
    special_tokens = args.special_tokens or DEFAULT_SPECIAL_TOKENS
    prepare_data(
        train_text_path=resolve_text_path(args.train_text),
        val_text_path=resolve_text_path(args.val_text),
        out_dir=args.out_dir,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
    )


if __name__ == "__main__":
    main()
