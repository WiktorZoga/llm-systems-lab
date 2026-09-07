import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

PARAMETERS = {
    "batch_size": ("benchmark", "batch_size"),
    "sequence_length": ("benchmark", "sequence_length"),
    "d_model": ("model", "d_model"),
    "num_layers": ("model", "num_layers"),
    "vocab_size": ("model", "vocab_size"),
    "num_heads": ("model", "num_heads"),
}

WORKLOADS = [
    "forward",
    "forward_backward",
    "optimizer_step",
]

def load_results(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_parameter_value(result: dict, parameter: str):
    section, name = PARAMETERS[parameter]
    return result["config"][section][name]

def collect_points(paths: list[Path], parameter: str):
    points = []
    reference = None

    for path in paths:
        result = load_results(path)

        # Everything except the sweep variable must describe the same workload.
        fixed_config = {
            section: dict(values) for section, values in result["config"].items()
        }
        fixed_config["model"].setdefault("attention_backend", "naive")
        section, name = PARAMETERS[parameter]
        del fixed_config[section][name]
        conditions = {
            "config": fixed_config,
            "environment": result["environment"],
            "optimizer": result.get("optimizer"),
            "measurement_notes": result.get("measurement_notes", "legacy"),
        }
        if reference is None:
            reference = conditions
        elif conditions != reference:
            raise ValueError(
                f"{path}: non-sweep settings differ. Compare one variable "
                "at a time on the same device and measurement procedure."
            )

        point = {
            "value": get_parameter_value(result, parameter),
            "parameter_count": result["parameter_count"],
            "measurements": result["measurements"],
        }

        points.append(point)

    points.sort(key=lambda point: point["value"])

    return points

def plot_results(points: list[dict], parameter: str, output_path: Path, statistic="mean"):
    x_values = [point["value"] for point in points]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    latency_axis = axes[0]
    throughput_axis = axes[1]
    parameters_axis = axes[2]

    for workload in WORKLOADS:
        latency_values = [
            point["measurements"][workload][f"{statistic}_seconds"] * 1000
            for point in points
        ]

        throughput_values = [
            point["measurements"][workload]["tokens_per_second"]
            for point in points
        ]

        latency_axis.plot(
            x_values,
            latency_values,
            marker="o",
            label=workload,
        )

        throughput_axis.plot(
            x_values,
            throughput_values,
            marker="o",
            label=workload,
        )

    parameter_counts = [
        point["parameter_count"]
        for point in points
    ]

    parameters_axis.plot(
        x_values,
        parameter_counts,
        marker="o",
    )

    latency_axis.set_title(f"{statistic.capitalize()} time")
    latency_axis.set_xlabel(parameter)
    latency_axis.set_ylabel("milliseconds")

    throughput_axis.set_title("Throughput (B*T / mean time)")
    throughput_axis.set_xlabel(parameter)
    throughput_axis.set_ylabel("tokens/s")

    parameters_axis.set_title("Parameter count")
    parameters_axis.set_xlabel(parameter)
    parameters_axis.set_ylabel("parameters")

    latency_axis.legend(loc="upper left")
    # throughput_axis.legend()

    fig.suptitle(f"Scaling experiment: {parameter}")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved plot to: {output_path}")


def main(args):
    paths = []

    for path in args.files:
        if path.is_absolute():
            paths.append(path)
        else:
            paths.append(ROOT / path)

    points = collect_points(paths, args.parameter)

    output_path = Path(args.output_dir) / f"{args.parameter}_{args.statistic}.png"

    if not output_path.is_absolute():
        output_path = ROOT / output_path

    plot_results(
        points=points,
        parameter=args.parameter,
        output_path=output_path,
        statistic=args.statistic,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--parameter", choices=PARAMETERS, required=True)
    parser.add_argument("--statistic", choices=["mean", "median"], default="mean")

    parser.add_argument("files", nargs="+", type=Path, help="Benchmark json files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/benchmarks/plots"),
        help="Directory for the generated plot.",
    )

    main(parser.parse_args())
