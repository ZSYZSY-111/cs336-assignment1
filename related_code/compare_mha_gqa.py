"""Compare MHA vs GQA training runs by reading their CSV logs.

Run after both train_model.py (MHA) and train_model_gqa.py (GQA) have finished.

Usage:
    python compare_mha_gqa.py
    python compare_mha_gqa.py --mha-log <path> --gqa-log <path> --plot
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_MHA_LOG = Path("/root/autodl-tmp/checkpoints_5m/train_log.csv")
DEFAULT_GQA_LOG = Path("/root/autodl-tmp/checkpoints_gqa/train_log_gqa.csv")


def _to_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def load_log(path: Path) -> dict:
    """Read a training CSV and return summary stats."""
    if not path.exists():
        raise FileNotFoundError(f"log not found: {path}")

    step_times = []
    tok_per_sec = []
    peak_mems = []
    final_train = None
    final_val = None

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ms = _to_float(row.get("ms_per_step"))
            tps = _to_float(row.get("tokens_per_sec"))
            mem = _to_float(row.get("peak_mem_mb"))
            tr = _to_float(row.get("train_loss"))
            va = _to_float(row.get("val_loss"))
            if ms is not None:
                step_times.append(ms)
            if tps is not None:
                tok_per_sec.append(tps)
            if mem is not None:
                peak_mems.append(mem)
            if tr is not None:
                final_train = tr
            if va is not None:
                final_val = va

    # skip first 10 steps for speed avg (warmup/compilation noise)
    warm = step_times[10:] if len(step_times) > 10 else step_times
    warm_tps = tok_per_sec[10:] if len(tok_per_sec) > 10 else tok_per_sec

    return {
        "avg_ms_per_step": sum(warm) / len(warm) if warm else 0.0,
        "avg_tokens_per_sec": sum(warm_tps) / len(warm_tps) if warm_tps else 0.0,
        "peak_mem_mb": max(peak_mems) if peak_mems else 0.0,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "num_steps": len(step_times),
    }


def fmt(v, suffix=""):
    return f"{v:.4f}{suffix}" if v is not None else "N/A"


def pct_change(base, new):
    """Percent change from base to new (negative = reduction)."""
    if base in (None, 0):
        return "N/A"
    return f"{(new - base) / base * 100:+.1f}%"


def print_comparison(mha: dict, gqa: dict) -> None:
    print("=" * 64)
    print(f"{'Metric':<22}{'MHA':>14}{'GQA':>14}{'Δ (GQA vs MHA)':>14}")
    print("-" * 64)

    rows = [
        ("final train loss", mha["final_train_loss"], gqa["final_train_loss"], False),
        ("final val loss", mha["final_val_loss"], gqa["final_val_loss"], False),
        ("avg ms/step", mha["avg_ms_per_step"], gqa["avg_ms_per_step"], True),
        ("avg tokens/sec", mha["avg_tokens_per_sec"], gqa["avg_tokens_per_sec"], True),
        ("peak mem (MB)", mha["peak_mem_mb"], gqa["peak_mem_mb"], True),
    ]
    for name, m, g, show_pct in rows:
        delta = pct_change(m, g) if show_pct else ""
        m_str = fmt(m) if m is not None else "N/A"
        g_str = fmt(g) if g is not None else "N/A"
        print(f"{name:<22}{m_str:>14}{g_str:>14}{delta:>14}")

    print("=" * 64)
    print(f"MHA steps: {mha['num_steps']} | GQA steps: {gqa['num_steps']}")
    print()
    print("Expected for GQA: lower mem, higher tok/s, similar loss.")


def maybe_plot(mha_log: Path, gqa_log: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot.")
        return

    def read_curve(path):
        its, vals = [], []
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                v = _to_float(row.get("val_loss"))
                if v is not None:
                    its.append(int(row["iteration"]))
                    vals.append(v)
        return its, vals

    mi, mv = read_curve(mha_log)
    gi, gv = read_curve(gqa_log)

    plt.figure(figsize=(8, 5))
    plt.plot(mi, mv, label="MHA", marker="o", markersize=3)
    plt.plot(gi, gv, label="GQA", marker="s", markersize=3)
    plt.xlabel("iteration")
    plt.ylabel("val loss")
    plt.title("MHA vs GQA — validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = Path("mha_vs_gqa_val_loss.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved plot to {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mha-log", type=Path, default=DEFAULT_MHA_LOG)
    parser.add_argument("--gqa-log", type=Path, default=DEFAULT_GQA_LOG)
    parser.add_argument("--plot", action="store_true", help="save a val-loss comparison plot")
    args = parser.parse_args()

    mha_stats = load_log(args.mha_log)
    gqa_stats = load_log(args.gqa_log)
    print_comparison(mha_stats, gqa_stats)

    if args.plot:
        maybe_plot(args.mha_log, args.gqa_log)
