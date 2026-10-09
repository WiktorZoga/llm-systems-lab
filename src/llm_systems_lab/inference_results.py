"""Read inference measurements for both notebooks and plot exports."""

import json
from os.path import commonpath
from pathlib import Path

import pandas as pd


def load_inference_results(paths):
    paths = [Path(path).resolve() for path in paths]
    # Keep enough of each path to distinguish repeats in different folders.
    base = Path(commonpath([path.parent for path in paths])) if paths else Path.cwd()
    runs, rows = {}, []
    for path in paths:
        name = str(path.relative_to(base).with_suffix(""))
        if name in runs:
            raise ValueError(f"Duplicate result file: {path}")
        result = json.loads(path.read_text())
        runs[name] = result
        settings = result["config"]["benchmark"]
        model = result["config"]["model"]
        rows.append({
            "run": name, "source_file": str(path), "mode": result["mode"],
            "device": result["environment"]["device"],
            "pytorch": result["environment"]["pytorch"],
            "dtype": settings["dtype"], "attention_backend": model["attention_backend"],
            "batch_size": settings["batch_size"], "prompt_length": settings["prompt_length"],
            "decode_steps": settings["decode_steps"], "seed": settings["seed"],
            "warmup": settings["warmup"], "iterations": settings["iterations"],
            "d_model": model["d_model"], "num_layers": model["num_layers"],
            "num_heads": model["num_heads"], "vocab_size": model["vocab_size"],
            "context_length": model["context_length"],
            "parameter_count": result["parameter_count"],
            "prefill_mean_ms": result["prefill"]["mean_ms"],
            "prefill_median_ms": result["prefill"]["median_ms"],
            "decode_mean_ms": result["decode"]["mean_ms"],
            "decode_median_ms": result["decode"]["median_ms"],
            "tokens_per_second": result["decode"]["tokens_per_second"],
            "cache_MiB": result["kv_cache_bytes"] / 2**20,
        })

    table = pd.DataFrame(rows)
    if rows:
        table = table.set_index("run")
    return runs, table
