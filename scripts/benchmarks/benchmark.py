import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.optim import AdamW

from llm_systems_lab.config import load_experiment_config
from llm_systems_lab.models.gpt import GPT

ROOT = Path(__file__).resolve().parents[1]

WORKLOADS = {
    "forward": "Training forward: full logits + cross_entropy, autograd enabled; no backward.",
    "forward_backward": "Same forward + loss.backward(); gradient reset outside timing.",
    "optimizer_step": "zero_grad(set_to_none=True) + same forward + backward + AdamW.step().",
}


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(config, warmup, iterations):

    device = torch.device(config.train.device)

    torch.manual_seed(config.train.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config.train.seed)

    # Explicit FP32
    torch.set_float32_matmul_precision("highest")

    batch_size = config.train.batch_size
    sequence_length = config.train.sequence_length

    tokens = torch.randint(config.model.vocab_size, (batch_size, sequence_length + 1))

    input_ids = tokens[:, :-1].contiguous().to(device)
    targets = tokens[:, 1:].contiguous().to(device)

    measurements = {}
    parameter_count = None

    for workload in WORKLOADS:
        # Each workload starts from the same weights and synthetic batch.
        model = GPT(config.model).to(device=device, dtype=torch.float32)
        model.train()
        optimizer = (
            AdamW(model.parameters(), lr=config.train.learning_rate)
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

            _, loss = model(input_ids, targets)
            if workload != "forward":
                loss.backward()
            if optimizer is not None:
                optimizer.step()

            synchronize(device)
            elapsed = time.perf_counter() - start

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

    return {
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else platform.processor() or platform.machine()
            ),
            "cuda_version": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "tokens_per_iteration": batch_size * sequence_length,
        "parameter_count": parameter_count,
        "optimizer": {
            "name": "AdamW", "lr": config.train.learning_rate,
            "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01,
            "foreach": False, "fused": False,
        },
        "measurements": measurements,
    }


def main(args):
    config_path = ROOT / "configs" / args.config
    config = load_experiment_config(config_path)

    result = run_benchmark(config, args.warmup, args.iterations)
    result["config_path"] = str(config_path.resolve())

    timestamp = datetime.now()

    result["created_at"] = timestamp.isoformat()

    output_dir = ROOT / "artifacts" / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (f"{config_path.stem}_{timestamp.strftime('%Y-%m-%d-%H-%M-%S')}.json")

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
    
    parser.add_argument("--config", default="scaling/baseline.toml",
                        help="Path relative to configs/, or an absolute path.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)

    main(parser.parse_args())
