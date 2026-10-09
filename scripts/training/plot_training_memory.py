"""Plot the saved-state and recomputation comparison, without running a model."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main(args):
    result = json.loads((ROOT / args.result).read_text())
    cases = result["cases"]
    names = [case["name"] for case in cases]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    values = [case["saved_activation_bytes"] / 2**20 for case in cases]
    bars = axes[0].bar(names, values, color=["#557b83", "#d1a45d"])
    axes[0].bar_label(bars, fmt="%.2f")
    axes[0].set(title="State saved for backward", ylabel="Unique activation storage [MiB]")
    axes[0].set_ylim(0, max(values) * 1.15)
    forward = [case["forward_block_calls"] for case in cases]
    backward = [case["backward_block_calls"] for case in cases]
    axes[1].bar(names, forward, label="Forward", color="#557b83")
    axes[1].bar(names, backward, bottom=forward, label="Recomputation in backward", color="#d1a45d")
    axes[1].set(title="Transformer block invocations", ylabel="Calls")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Checkpointing · {result['device']} · {result['dtype']}")
    fig.tight_layout()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
