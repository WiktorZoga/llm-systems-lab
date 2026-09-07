import argparse
import json
import statistics
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.optim import AdamW

from llm_systems_lab.config import load_benchmark_config
from llm_systems_lab.device import select_device, synchronize
from llm_systems_lab.models.gpt import GPT

ROOT = Path(__file__).resolve().parents[2]

WORKLOADS = {
    "forward": "Training forward: full logits + cross_entropy, autograd enabled; no backward.",
    "forward_backward": "Same forward + loss.backward(); gradient reset outside timing.",
    "optimizer_step": "zero_grad(set_to_none=True) + same forward + backward + AdamW.step().",
}


def run_benchmark(config):

    device = select_device(config.device)

    torch.manual_seed(config.seed)

    # Explicit FP32
    torch.set_float32_matmul_precision("highest")

    batch_size = config.batch_size
    sequence_length = config.sequence_length

    tokens = torch.randint(config.model.vocab_size, (batch_size, sequence_length + 1))

    input_ids = tokens[:, :-1].contiguous().to(device)
    targets = tokens[:, 1:].contiguous().to(device)

    measurements = {}
    parameter_count = None
    warmup = config.warmup_iterations
    iterations = config.iterations

    for workload in WORKLOADS:
        # Each workload starts from the same weights and synthetic batch.
        torch.manual_seed(config.seed)
        model = GPT(config.model).to(device=device, dtype=torch.float32)
        model.train()
        optimizer = (
            AdamW(
                model.parameters(), lr=config.learning_rate,
                foreach=False, fused=False,
            )
            if workload == "optimizer_step" else None
        )
        parameter_count = sum(p.numel() for p in model.parameters())
        samples = []

        for index in range(warmup + iterations):
            if workload != "optimizer_step":
                model.zero_grad(set_to_none=True)

            synchronize(device)
            start = time.perf_counter()

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits, loss = model(input_ids, targets)
            if workload != "forward":
                loss.backward()
            if optimizer is not None:
                optimizer.step()

            synchronize(device)
            elapsed = time.perf_counter() - start

            # Python evaluates the next model(...) before replacing logits/loss.
            # Remove these references now so the old outputs (and saved tensors
            # for forward-only) do not stay alive during the next forward.
            # This happens after timing; PyTorch may keep freed memory in its cache.
            del logits, loss

            if index >= warmup:
                samples.append(elapsed)

        mean_seconds = statistics.mean(samples)
        median_seconds = statistics.median(samples)
        measurements[workload] = {
            "description": WORKLOADS[workload],
            "samples_seconds": samples,
            "mean_seconds": mean_seconds,
            "median_seconds": median_seconds,
            "tokens_per_second": batch_size * sequence_length / mean_seconds,
        }
        # This workload is finished. Release its model and optimizer state
        # before constructing a fresh model for the next workload.
        del optimizer, model

    return {
        "config": {
            "model": asdict(config.model),
            "benchmark": {
                "batch_size": config.batch_size,
                "sequence_length": config.sequence_length,
                "warmup_iterations": config.warmup_iterations,
                "iterations": config.iterations,
                "learning_rate": config.learning_rate,
                "seed": config.seed,
                "device": config.device,
                "dtype": config.dtype,
            },
        },
        "environment": {"device": str(device), "pytorch": torch.__version__},
        "measurement_notes": (
            "Outputs released after each timing; AdamW foreach=False, fused=False. "
            "Allocator caches stay warm. Compare within this measurement procedure."
        ),
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "tokens_per_iteration": batch_size * sequence_length,
        "parameter_count": parameter_count,
        "optimizer": {
            "name": "AdamW", "lr": config.learning_rate,
            "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01,
            "foreach": False, "fused": False,
        },
        "measurements": measurements,
    }


def main(args):
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config = load_benchmark_config(config_path)
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.attention_backend is not None:
        config = replace(
            config, model=replace(config.model, attention_backend=args.attention_backend),
        )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_path = output_dir / f"{config_path.stem}.json"
    if output_path.exists():
        raise FileExistsError(f"Result already exists: {output_path}; use a new output directory")

    result = run_benchmark(config)
    result["config_path"] = str(config_path.resolve())

    timestamp = datetime.now(timezone.utc)

    result["created_at"] = timestamp.isoformat()

    output_dir.mkdir(parents=True, exist_ok=True)

    with output_path.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")

    print(f"Parameters: {result['parameter_count']:,}")
    
    for name, measurement in result["measurements"].items():
        print(
            f"{name}: mean={measurement['mean_seconds'] * 1000:.3f} ms | "
            f"median={measurement['median_seconds'] * 1000:.3f} ms | "
            f"tokens/s={measurement['tokens_per_second']:.2f}"
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", default="configs/benchmarks/scaling/baseline.toml",
                        help="Path relative or an absolute path to your config file.")
    parser.add_argument("--output-dir", default="artifacts/benchmarks/manual",
                        help="Directory for the JSON result.")
    parser.add_argument("--device", help="Override config device: cpu, mps, cuda or cuda:N.")
    parser.add_argument("--attention-backend", choices=["naive", "sdpa"],
                        help="Override only the attention implementation for an A/B run.")

    main(parser.parse_args())
