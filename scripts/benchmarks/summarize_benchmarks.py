import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return ROOT / path


def load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def result_to_rows(path: Path) -> list[dict]:
    result = load_result(path)

    model_config = result["config"]["model"]
    benchmark_config = result["config"]["benchmark"]

    common_values = {
        "config": path.stem,
        "device": result["environment"]["device"],
        "batch_size": benchmark_config["batch_size"],
        "sequence_length": benchmark_config["sequence_length"],
        "d_model": model_config["d_model"],
        "num_layers": model_config["num_layers"],
        "num_heads": model_config["num_heads"],
        "vocab_size": model_config["vocab_size"],
        "parameter_count": result["parameter_count"],
        "tokens_per_iteration": result["tokens_per_iteration"],
    }

    rows = []

    for workload, measurement in result["measurements"].items():
        row = {
            **common_values,
            "workload": workload,
            "mean_ms": measurement["mean_seconds"] * 1000,
            "median_ms": measurement["median_seconds"] * 1000,
            "tokens_per_second": measurement["tokens_per_second"],
        }

        rows.append(row)

    return rows


def main(args):
    input_dir = resolve_path(args.input_dir)

    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No benchmark JSON files found in {input_dir}"
        )

    rows = []

    for json_file in json_files:
        rows.extend(result_to_rows(json_file))

    dataframe = pd.DataFrame(rows)

    dataframe = dataframe.sort_values(
        ["workload", "config"]
    ).reset_index(drop=True)

    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else input_dir / "tables"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "summary.csv"

    dataframe.to_csv(output_path, index=False)

    print(dataframe.to_string(index=False))
    print(f"\nSaved table to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing benchmark JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the generated CSV table.",
    )

    main(parser.parse_args())