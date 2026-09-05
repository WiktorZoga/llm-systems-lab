import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

from llm_systems_lab.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = load_experiment_config(ROOT / "configs" / "gpt_tiny_smoke.toml")

    OUTPUT_DIR = ROOT / config.run.output_dir / config.run.name
    metrics_path = OUTPUT_DIR / "metrics.jsonl" 

    CHARTS_DIR = OUTPUT_DIR / "charts"

    os.makedirs(CHARTS_DIR, exist_ok=True)

    metrics = []

    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))

    steps = [metric["step"] for metric in metrics]
    train_losses = [metric["train_loss"] for metric in metrics]
    tokens_per_second = [metric["tokens/s"] for metric in metrics]
    # step_times = [metric["step_time_sec"] for metric in metrics]
    # lr = [metric["lr"] for metric in metrics]

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
        CHARTS_DIR / "loss.png",
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
        CHARTS_DIR / "throughput.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


if __name__ == "__main__":
    main()