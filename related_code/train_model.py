from pathlib import Path
import csv
import pickle
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent

CONFIG = {
    "text_path": Path("/root/autodl-tmp/TinyStoriesV2-GPT4-train.txt"),
    "val_text_path": Path("/root/autodl-tmp/TinyStoriesV2-GPT4-valid.txt"),
    "prepared_data_dir": Path("/root/autodl-tmp/prepared_data"),
    "use_prepared_data": True,
    "vocab_size": 512,
    "special_tokens": ["<|endoftext|>"],
    "train_fraction": 0.9,
    "device": "cuda",
    "context_length": 256,
    "batch_size": 32,
    "num_iters": 20000,
    "eval_interval": 500,
    "eval_iters": 20,
    "d_model": 512,
    "num_layers": 6,
    "num_heads": 8,
    "d_ff": 2048,
    "rope_theta": 10000.0,
    "learning_rate": 3e-4,
    "min_learning_rate": 3e-5,
    "warmup_iters": 1000,
    "cosine_cycle_iters": 20000,
    "weight_decay": 0.01,
    "max_l2_norm": 1.0,
    "checkpoint_dir": Path("/root/autodl-tmp/checkpoints_5m"),
    "checkpoint_interval": 1000,
    "log_path": None,
}

from data import get_batch
from nn_util import cross_entropy, gradient_clipping
from optimizer import AdamW, get_lr_cosine_schedule
from serialization import save_checkpoint
from train_bpe_upgrade import decode, encode, run_train_bpe
from train_transformer import TransformerLM


def compute_loss(model, dataset, batch_size, context_length, vocab_size, device):
    x, y = get_batch(dataset, batch_size, context_length, device)
    logits = model(x)
    return cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))


def estimate_loss(model, train_data, val_data, batch_size, context_length, vocab_size, device, eval_iters):
    was_training = model.training
    model.eval()
    losses = {}
    with torch.no_grad():
        for split, dataset in [("train", train_data), ("val", val_data)]:
            split_losses = []
            for _ in range(eval_iters):
                loss = compute_loss(model, dataset, batch_size, context_length, vocab_size, device)
                split_losses.append(loss.item())
            losses[split] = sum(split_losses) / len(split_losses)
    if was_training:
        model.train()
    return losses


def load_prepared_data(prepared_data_dir):
    prepared_data_dir = Path(prepared_data_dir)
    tokenizer_path = prepared_data_dir / "tokenizer.pkl"
    train_tokens_path = prepared_data_dir / "train_tokens.npy"
    val_tokens_path = prepared_data_dir / "val_tokens.npy"

    missing_paths = [
        path
        for path in [tokenizer_path, train_tokens_path, val_tokens_path]
        if not path.exists()
    ]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"missing prepared data files: {missing}. "
            "Run prepare_data.py first, or set use_prepared_data=False."
        )

    with tokenizer_path.open("rb") as f:
        tokenizer = pickle.load(f)

    train_data = np.load(train_tokens_path, mmap_mode="r")
    val_data = np.load(val_tokens_path, mmap_mode="r")

    return tokenizer["vocab"], tokenizer["merges"], train_data, val_data


def main():
    config = CONFIG
    config["checkpoint_dir"].mkdir(exist_ok=True)
    log_path = Path(config["log_path"]) if config["log_path"] is not None else config["checkpoint_dir"] / "train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if config["use_prepared_data"]:
        print(f"loading prepared data from {config['prepared_data_dir']}...", flush=True)
        vocab, merges, train_data, val_data = load_prepared_data(config["prepared_data_dir"])
        print("finished loading prepared data.", flush=True)
    else:
        print(f"training BPE on {config['text_path']} with vocab_size={config['vocab_size']}...", flush=True)
        vocab, merges = run_train_bpe(config["text_path"], config["vocab_size"], config["special_tokens"])
        print("finished BPE training; encoding train and val text...", flush=True)
        train_text = config["text_path"].read_text(encoding="utf-8")
        val_text = config["val_text_path"].read_text(encoding="utf-8")
        train_token_ids = encode(train_text, vocab, merges, config["special_tokens"])
        val_token_ids = encode(val_text, vocab, merges, config["special_tokens"])
        train_data = np.array(train_token_ids, dtype=np.int64)
        val_data = np.array(val_token_ids, dtype=np.int64)
        print("finished encoding train and val text.", flush=True)

    if config["use_prepared_data"]:
        print(f"prepared data dir: {config['prepared_data_dir']}")
    else:
        print(f"train text path: {config['text_path']}")
        print(f"val text path: {config['val_text_path']}")
    print(f"vocab size: {len(vocab)}")
    print(f"num merges: {len(merges)}")
    print(f"train tokens: {len(train_data)}")
    print(f"val tokens: {len(val_data)}")
    print(f"first 20 train ids: {train_data[:20].tolist()}")
    print("decoded train preview:")
    print(decode(train_data[:80].tolist(), vocab))

    model = TransformerLM(
        vocab_size=len(vocab),
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=config["device"],
        dtype=torch.float32,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    print(f"logging training metrics to {log_path}", flush=True)

    use_cuda = torch.cuda.is_available() and str(config["device"]).startswith("cuda")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    with log_path.open("w", newline="") as log_file:
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "iteration",
                "lr",
                "step_loss",
                "train_loss",
                "val_loss",
                "ms_per_step",
                "tokens_per_sec",
                "peak_mem_mb",
            ],
        )
        log_writer.writeheader()

        model.train()
        for it in range(config["num_iters"]):
            lr = get_lr_cosine_schedule(
                it,
                config["learning_rate"],
                config["min_learning_rate"],
                config["warmup_iters"],
                config["cosine_cycle_iters"],
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            eval_losses = None
            if it % config["eval_interval"] == 0:
                eval_losses = estimate_loss(
                    model,
                    train_data,
                    val_data,
                    config["batch_size"],
                    config["context_length"],
                    len(vocab),
                    config["device"],
                    config["eval_iters"],
                )
                print(f"iter {it:02d} | lr {lr:.6f} | train {eval_losses['train']:.4f} | val {eval_losses['val']:.4f}")

            x, y = get_batch(
                train_data,
                config["batch_size"],
                config["context_length"],
                config["device"],
            )

            if use_cuda:
                torch.cuda.synchronize()
            step_start = time.perf_counter()

            optimizer.zero_grad()
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, len(vocab)), y.reshape(-1))
            loss.backward()
            gradient_clipping(model.parameters(), max_l2_norm=config["max_l2_norm"])
            optimizer.step()

            if use_cuda:
                torch.cuda.synchronize()
            step_time = time.perf_counter() - step_start

            ms_per_step = step_time * 1000.0
            tokens_per_step = config["batch_size"] * config["context_length"]
            tokens_per_sec = tokens_per_step / step_time if step_time > 0 else 0.0
            peak_mem_mb = (
                torch.cuda.max_memory_allocated() / (1024**2) if use_cuda else 0.0
            )

            step_loss = loss.item()
            print(
                f"iter {it:02d} | lr {lr:.6f} | loss {step_loss:.4f} | "
                f"{ms_per_step:.1f}ms/step | {tokens_per_sec / 1000:.0f}k tok/s | "
                f"mem {peak_mem_mb:.0f}MB"
            )
            log_writer.writerow(
                {
                    "iteration": it,
                    "lr": lr,
                    "step_loss": step_loss,
                    "train_loss": "" if eval_losses is None else eval_losses["train"],
                    "val_loss": "" if eval_losses is None else eval_losses["val"],
                    "ms_per_step": f"{ms_per_step:.3f}",
                    "tokens_per_sec": f"{tokens_per_sec:.1f}",
                    "peak_mem_mb": f"{peak_mem_mb:.2f}",
                }
            )
            log_file.flush()

            if (it + 1) % config["checkpoint_interval"] == 0:
                checkpoint_path = config["checkpoint_dir"] / f"tiny_step_{it + 1}.pt"
                save_checkpoint(model, optimizer, it + 1, checkpoint_path)
                print(f"saved checkpoint: {checkpoint_path}")

    losses = estimate_loss(
        model,
        train_data,
        val_data,
        config["batch_size"],
        config["context_length"],
        len(vocab),
        config["device"],
        config["eval_iters"],
    )
    print(f"final | train {losses['train']:.4f} | val {losses['val']:.4f}")

    if use_cuda:
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"peak GPU memory: {peak_mem_mb:.2f} MB")

    final_checkpoint_path = config["checkpoint_dir"] / "tiny_final.pt"
    save_checkpoint(model, optimizer, config["num_iters"], final_checkpoint_path)
    print(f"saved checkpoint: {final_checkpoint_path}")


if __name__ == "__main__":
    main()
