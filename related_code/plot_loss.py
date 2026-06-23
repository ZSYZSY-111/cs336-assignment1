from pathlib import Path
import argparse
import csv
import sys


ROOT = Path(__file__).resolve().parent


def _read_float(value):
    if value == "" or value is None:
        return None
    return float(value)


def load_loss_log(csv_path):
    csv_path = Path(csv_path)
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "iteration": int(row["iteration"]),
                    "lr": _read_float(row.get("lr")),
                    "step_loss": _read_float(row.get("step_loss")),
                    "train_loss": _read_float(row.get("train_loss")),
                    "val_loss": _read_float(row.get("val_loss")),
                }
            )
    return rows


def plot_loss(csv_path, output_path=None, show=False, max_step=None):
    """Plot step, train, and validation loss from a training log CSV."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plot_loss(). Install it with: "
            "python3 -m pip install matplotlib"
        ) from exc

    csv_path = Path(csv_path)
    if output_path is None:
        output_path = csv_path.with_name(csv_path.stem + "_loss.png")
    output_path = Path(output_path)

    rows = load_loss_log(csv_path)
    if not rows:
        raise ValueError(f"no rows found in {csv_path}")
    if max_step is not None:
        rows = [row for row in rows if row["iteration"] <= max_step]
        if not rows:
            raise ValueError(f"no rows at or before step {max_step} in {csv_path}")

    iterations = [row["iteration"] for row in rows]
    step_losses = [row["step_loss"] for row in rows]

    eval_rows = [
        row
        for row in rows
        if row["train_loss"] is not None or row["val_loss"] is not None
    ]
    eval_iterations = [row["iteration"] for row in eval_rows]
    train_losses = [row["train_loss"] for row in eval_rows]
    val_losses = [row["val_loss"] for row in eval_rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, step_losses, label="step loss", alpha=0.35, linewidth=1)

    if any(loss is not None for loss in train_losses):
        ax.plot(
            eval_iterations,
            train_losses,
            label="train loss",
            marker="o",
            linewidth=2,
        )
    if any(loss is not None for loss in val_losses):
        ax.plot(
            eval_iterations,
            val_losses,
            label="validation loss",
            marker="o",
            linewidth=2,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    title = "Training Loss"
    if max_step is not None:
        title += f" (First {max_step} Steps)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot loss curves from train_log.csv.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=ROOT / "train_log.csv",
        help="Path to the training log CSV.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to save the PNG. Defaults to <csv_stem>_loss.png.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Only plot rows with iteration <= this step. Defaults to all steps.",
    )
    args = parser.parse_args()

    try:
        output_path = plot_loss(args.csv_path, args.output, args.show, args.max_step)
    except ModuleNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"saved loss plot to {output_path}")


if __name__ == "__main__":
    main()
