from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import torch

from train_bpe_upgrade import decode, encode, run_train_bpe
from train_transformer import TransformerLM


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "tiny_final.pt"
DEFAULT_TOKENIZER = ROOT / "prepared_data" / "tokenizer.pkl"
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]


def load_tokenizer(
    tokenizer_path: Path,
    train_text: Path | None,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]], list[str]]:
    if tokenizer_path.exists():
        with tokenizer_path.open("rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer["vocab"], tokenizer["merges"], tokenizer.get("special_tokens", special_tokens)

    if train_text is None:
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Run prepare_data.py first, or pass --train-text to rebuild the BPE tokenizer."
        )

    vocab, merges = run_train_bpe(train_text, vocab_size, special_tokens)
    return vocab, merges, special_tokens


def infer_model_config(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    vocab_size, d_model = state_dict["token_embeddings.weight"].shape
    d_ff = state_dict["layers.0.ffn.w1.weight"].shape[0]
    layer_ids = {
        int(match.group(1))
        for key in state_dict
        if (match := re.match(r"layers\.(\d+)\.", key))
    }

    return {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "d_ff": d_ff,
        "num_layers": max(layer_ids) + 1,
    }


def choose_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None:
        top_values, _ = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
        logits = logits.masked_fill(logits < top_values[:, [-1]], float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: TransformerLM,
    input_ids: list[int],
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    top_k: int | None,
    stop_token_id: int | None,
    device: torch.device,
) -> list[int]:
    generated = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = generated[:, -context_length:]
        logits = model(context)[:, -1, :]
        next_token = choose_next_token(logits, temperature=temperature, top_k=top_k)
        generated = torch.cat([generated, next_token], dim=1)

        if stop_token_id is not None and next_token.item() == stop_token_id:
            break

    return generated[0].tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained CS336 TransformerLM checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--train-text", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--special-token", action="append", dest="special_tokens", default=None)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    config = infer_model_config(state_dict)

    special_tokens = args.special_tokens or DEFAULT_SPECIAL_TOKENS
    vocab, merges, special_tokens = load_tokenizer(
        tokenizer_path=args.tokenizer_path,
        train_text=args.train_text,
        vocab_size=config["vocab_size"],
        special_tokens=special_tokens,
    )
    if len(vocab) != config["vocab_size"]:
        raise ValueError(
            f"Tokenizer vocab size ({len(vocab)}) does not match checkpoint vocab size "
            f"({config['vocab_size']}). Use the tokenizer from the same training run."
        )

    stop_token_id = None
    if special_tokens:
        token_to_id = {token: token_id for token_id, token in vocab.items()}
        stop_token_id = token_to_id.get(special_tokens[0].encode("utf-8"))

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=args.context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=args.num_heads,
        d_ff=config["d_ff"],
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32,
    )
    model.load_state_dict(state_dict)
    model.eval()

    input_ids = encode(args.prompt, vocab, merges, special_tokens)
    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        temperature=args.temperature,
        top_k=args.top_k,
        stop_token_id=stop_token_id,
        device=device,
    )
    print(decode(output_ids, vocab))


if __name__ == "__main__":
    main()
