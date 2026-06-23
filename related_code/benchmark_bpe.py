"""Benchmark: single-process BPE vs parallel BPE.

Usage:
    python benchmark_bpe.py                          # uses TinyStories train by default
    python benchmark_bpe.py --input data/owt_train.txt --vocab-size 32000
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT = ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
DEFAULT_VOCAB_SIZE = 512
SPECIAL_TOKENS = ["<|endoftext|>"]


def run_benchmark(input_path: Path, vocab_size: int) -> None:
    from train_bpe_upgrade import run_train_bpe as run_single
    from train_bpe_parallel import run_train_bpe as run_parallel
    import os

    num_cores = os.cpu_count()
    file_mb = input_path.stat().st_size / (1024 ** 2)

    print(f"Input file : {input_path.name} ({file_mb:.1f} MB)")
    print(f"vocab_size : {vocab_size}")
    print(f"CPU cores  : {num_cores}")
    print("-" * 48)

    # --- single process ---
    print("Running single-process BPE...", flush=True)
    t0 = time.perf_counter()
    vocab_s, merges_s = run_single(input_path, vocab_size, SPECIAL_TOKENS)
    t_single = time.perf_counter() - t0
    print(f"  done in {t_single:.2f}s")

    # --- parallel ---
    print(f"Running parallel BPE ({num_cores} processes)...", flush=True)
    t0 = time.perf_counter()
    vocab_p, merges_p = run_parallel(input_path, vocab_size, SPECIAL_TOKENS)
    t_parallel = time.perf_counter() - t0
    print(f"  done in {t_parallel:.2f}s")

    # --- verify results match ---
    merges_match = merges_s == merges_p
    vocab_match = vocab_s == vocab_p

    print()
    print("=" * 48)
    print(f"  Single-process : {t_single:8.2f}s")
    print(f"  Parallel       : {t_parallel:8.2f}s")
    print(f"  Speedup        : {t_single / t_parallel:8.2f}x")
    print(f"  Results match  : merges={merges_match}, vocab={vocab_match}")
    print("=" * 48)

    if not merges_match or not vocab_match:
        print("WARNING: results do not match — check parallel implementation!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    args = parser.parse_args()
    run_benchmark(args.input, args.vocab_size)
