"""Draw prefill/decode results and save PNGs plus a CSV table."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from llm_systems_lab.inference_results import load_inference_results

ROOT = Path(__file__).resolve().parents[2]


def main(args):
    runs, table = load_inference_results([ROOT / path for path in args.files])
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "summary.csv")
    columns = ["prefill_mean_ms", "prefill_median_ms", "decode_mean_ms",
               "decode_median_ms", "tokens_per_second", "cache_MiB"]
    print(table[columns].round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    table[["prefill_mean_ms", "prefill_median_ms"]].plot.bar(
        ax=axes[0], rot=35, fontsize=8,
    )
    axes[0].set(title="Prefill", ylabel="Time [ms]", xlabel="")
    axes[0].legend(["Mean", "Median"])
    for name, result in runs.items():
        samples = pd.DataFrame(result["decode"]["samples_seconds"]) * 1000
        lengths = result["decode"]["context_lengths"]
        line, = axes[1].plot(lengths, samples.mean(), label=name)
        axes[1].plot(lengths, samples.median(), "--", color=line.get_color())
    axes[1].set(
        title="Decode: mean (solid), median (dashed)",
        xlabel="Context length [tokens]", ylabel="Time per batch step [ms]",
    )
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "latency.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    table["tokens_per_second"].plot.bar(ax=ax, rot=35, fontsize=8)
    ax.set(title="Decode throughput", ylabel="Tokens/s across batch", xlabel="")
    fig.tight_layout()
    fig.savefig(output / "throughput.png", dpi=150)
    plt.close(fig)
    print(f"Saved figures and summary to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="JSONs from inference.py")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmarks/inference/figures"))
    main(parser.parse_args())
