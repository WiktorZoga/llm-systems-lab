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


def memory_snapshot(stage, device, step=0):
    synchronize(device)
    row = {"step": step, "stage": stage}
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


def measure_memory(config, forward_no_grad=False, workload="optimizer_step"):
    if forward_no_grad:
        workload = "forward"
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

    optimizer = None
    if workload == "optimizer_step":
        optimizer = AdamW(
            model.parameters(), lr=config.learning_rate, foreach=False, fused=False,
        )
        snapshots.append(memory_snapshot("after_optimizer_creation", device))

    tokens = torch.randint(
        config.model.vocab_size, (config.batch_size, config.sequence_length + 1),
    )
    input_ids = tokens[:, :-1].contiguous().to(device)
    targets = tokens[:, 1:].contiguous().to(device)
    for step in range(1, config.iterations + 1):
        model.zero_grad(set_to_none=True)
        snapshots.append(memory_snapshot("before_forward", device, step))

        with torch.set_grad_enabled(not forward_no_grad):
            logits, loss = model(input_ids, targets)
        snapshots.append(memory_snapshot("after_forward", device, step))

        if workload != "forward":
            loss.backward()
            snapshots.append(memory_snapshot("after_backward", device, step))
        if optimizer is not None:
            optimizer.step()
            snapshots.append(memory_snapshot("after_optimizer_step", device, step))

        # Keep only the measurements, not tensors or graphs from earlier steps.
        del logits, loss
        snapshots.append(memory_snapshot("after_release", device, step))
        if step == 1 or step % 100 == 0 or step == config.iterations:
            print(f"Memory: {workload}, step {step}/{config.iterations}", flush=True)

    return {
        "forward_no_grad": forward_no_grad,
        "workload": workload,
        "config": asdict(config),
        "environment": {"device": str(device), "pytorch": torch.__version__},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "buffer_bytes": sum(b.numel() * b.element_size() for b in model.buffers()),
        "optimizer": ({"name": "AdamW", "lr": config.learning_rate,
                       "foreach": False, "fused": False} if optimizer is not None else None),
        "notes": (
            "Bytes, synchronized snapshots for config.iterations steps, no discarded warmup. "
            "Synthetic batch created once before the loop. Gradients cleared before forward. "
            "Logits/loss retained until after_release; gradients remain until the next step. "
            "AdamW state is allocated on the first update and reused. No empty_cache calls. "
            "With forward_no_grad, only forward is run. Extra synchronization changes execution; "
            "these are not latency measurements or samples from the separate timing runs. "
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

    result = measure_memory(
        config, forward_no_grad=args.forward_no_grad, workload=args.workload,
    )
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["config_path"] = str(config_path.resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")

    print(f"Saved {len(result['snapshots'])} memory snapshots.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmarks/scaling/baseline.toml")
    parser.add_argument("--device", help="Override device: cpu, mps, cuda or cuda:N")
    parser.add_argument("--attention-backend", choices=["naive", "sdpa"])
    parser.add_argument("--forward-no-grad", action="store_true",
                        help="Run only forward without autograd, regardless of --workload.")
    parser.add_argument("--workload", default="optimizer_step",
                        choices=["forward", "forward_backward", "optimizer_step"],
                        help="Workload repeated config.iterations times; default: full step.")
    parser.add_argument("--output", default="artifacts/memory/baseline.json")
    main(parser.parse_args())
