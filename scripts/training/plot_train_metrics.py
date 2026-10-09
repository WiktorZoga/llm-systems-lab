import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from llm_systems_lab.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[2]


def main(args):
    config = load_experiment_config(ROOT / "configs" / args.config)

    run_dir = ROOT / config.run.output_dir / config.run.name
    metrics_path = ROOT / args.metrics if args.metrics else run_dir / "metrics.jsonl"

    charts_dir = ROOT / args.output_dir if args.output_dir else run_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    metrics = []

    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))

    steps = [metric["step"] for metric in metrics]
    train_losses = [metric["train_loss"] for metric in metrics]
    tokens_per_second = [metric["tokens/s"] for metric in metrics]

    val_metrics = [
        metric for metric in metrics
        if "val_loss" in metric
    ]

    val_steps = [metric["step"] for metric in val_metrics]
    val_losses = [metric["val_loss"] for metric in val_metrics]

    fig, ax = plt.subplots()
    ax.plot(steps, train_losses, label="train loss")
    ax.plot(val_steps, val_losses, label="val loss", marker="o")

    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.grid(True)

    fig.savefig(
        charts_dir / "loss.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    fig, ax = plt.subplots()

    ax.plot(steps, tokens_per_second)

    ax.set_xlabel("steps")
    ax.set_ylabel("tokens / second")
    ax.set_title("Training throughput")
    ax.grid(True)

    fig.savefig(
        charts_dir / "throughput.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="gpt_tiny_smoke.toml")
    parser.add_argument("--metrics", type=Path, help="Plot an already saved metrics.jsonl file.")
    parser.add_argument("--output-dir", type=Path)
    main(parser.parse_args())
