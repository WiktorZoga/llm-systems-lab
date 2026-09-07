"""Six synchronized snapshots of one training step; no latency benchmark."""

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.optim import AdamW

from llm_systems_lab.config import load_benchmark_config
from llm_systems_lab.device import select_device, synchronize
from llm_systems_lab.models.gpt import GPT

ROOT = Path(__file__).resolve().parents[2]


def memory_snapshot(stage, device):
    synchronize(device)
    row = {"stage": stage}
    if device.type == "mps":
        row["tensor_bytes"] = torch.mps.current_allocated_memory()
        row["driver_bytes"] = torch.mps.driver_allocated_memory()
        # MPS exposes these snapshots, not a CUDA-style allocator peak here.
        row["peak_tensor_bytes"] = None
    elif device.type == "cuda":
        row["tensor_bytes"] = torch.cuda.memory_allocated(device)
        row["reserved_bytes"] = torch.cuda.memory_reserved(device)
        row["peak_tensor_bytes"] = torch.cuda.max_memory_allocated(device)
    else:
        # CPU allocator measurements are outside this utility's scope.
        row["tensor_bytes"] = None
        row["peak_tensor_bytes"] = None
    return row


def measure_memory(config):
    device = select_device(config.device)
    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    snapshots = [memory_snapshot("before_model", device)]
    model = GPT(config.model).to(device=device, dtype=torch.float32)
    model.train()
    snapshots.append(memory_snapshot("after_model", device))

    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, foreach=False, fused=False,
    )
    snapshots.append(memory_snapshot("after_optimizer_creation", device))

    tokens = torch.randint(
        config.model.vocab_size, (config.batch_size, config.sequence_length + 1),
    )
    input_ids = tokens[:, :-1].contiguous().to(device)
    targets = tokens[:, 1:].contiguous().to(device)
    logits, loss = model(input_ids, targets)
    snapshots.append(memory_snapshot("after_forward", device))
    loss.backward()
    snapshots.append(memory_snapshot("after_backward", device))
    optimizer.step()
    snapshots.append(memory_snapshot("after_first_optimizer_step", device))

    return {
        "config": asdict(config),
        "environment": {"device": str(device), "pytorch": torch.__version__},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "retained_logits_shape": list(logits.shape),
        "optimizer": {"name": "AdamW", "lr": config.learning_rate,
                      "foreach": False, "fused": False},
        "notes": (
            "Bytes, synchronized snapshots of a first training step, no warmup. "
            "Batch allocated after optimizer creation; logits/loss remain alive. "
            "Gradients remain after step. AdamW state is allocated lazily on step. "
            "MPS driver bytes include caches/framework allocations, not just tensors. "
            "CUDA peak is cumulative since reset before model creation. "
            "CPU values are unavailable (null). Device counters exclude CPU copies."
        ),
        "snapshots": snapshots,
    }


def main(args):
    config_path = ROOT / args.config
    config = load_benchmark_config(config_path)
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.attention_backend is not None:
        config = replace(
            config, model=replace(config.model, attention_backend=args.attention_backend),
        )
    output_path = ROOT / args.output
    if output_path.exists():
        raise FileExistsError(f"Result already exists: {output_path}")

    result = measure_memory(config)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["config_path"] = str(config_path.resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")

    for row in result["snapshots"]:
        print(row)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmarks/scaling/baseline.toml")
    parser.add_argument("--device", help="Override device: cpu, mps, cuda or cuda:N")
    parser.add_argument("--attention-backend", choices=["naive", "sdpa"])
    parser.add_argument("--output", default="artifacts/memory/baseline.json")
    main(parser.parse_args())
