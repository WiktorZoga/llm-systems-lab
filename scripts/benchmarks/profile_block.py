"""Time and profile one Transformer block on synthetic input."""

import argparse
import json
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from llm_systems_lab.config import load_benchmark_config
from llm_systems_lab.device import select_device, synchronize
from llm_systems_lab.models.gpt import Block

ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", help="Override config device: cpu, mps, cuda or cuda:N")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"])
    parser.add_argument("--attention-backend", choices=["naive", "sdpa"])
    parser.add_argument("--workload", required=True, choices=["forward", "forward_backward"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace", help="Optional Chrome trace path. Profiles three iterations.")
    return parser.parse_args(argv)


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def reset_gradients(block, x):
    block.zero_grad(set_to_none=True)
    x.grad = None


def run_once(block, x, grad_output, backward):
    output = block(x)
    if backward:
        output.backward(grad_output)
    return output


def validate(block, x, grad_output, backward):
    reset_gradients(block, x)
    output = run_once(block, x, grad_output, backward)
    result = {
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all().item()),
        "gradients_finite": None,
    }
    if backward:
        gradients = [x.grad, *(parameter.grad for parameter in block.parameters())]
        result["gradients_finite"] = all(
            gradient is not None and bool(torch.isfinite(gradient).all().item())
            for gradient in gradients
        )
    del output
    reset_gradients(block, x)
    return result


def measure(block, x, grad_output, backward, device, warmup, iterations):
    samples = []
    for index in range(warmup + iterations):
        reset_gradients(block, x)
        synchronize(device)
        start = time.perf_counter()
        output = run_once(block, x, grad_output, backward)
        synchronize(device)
        elapsed = time.perf_counter() - start
        del output
        if index >= warmup:
            samples.append(elapsed)
    reset_gradients(block, x)
    return samples


def profile_operations(block, x, grad_output, backward, device, trace_path):
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
        for _ in range(3):
            reset_gradients(block, x)
            output = run_once(block, x, grad_output, backward)
            synchronize(device)
            del output
            profiler.step()

    reset_gradients(block, x)
    profiler.export_chrome_trace(str(trace_path))
    rows = []
    for event in profiler.key_averages(group_by_input_shape=True):
        rows.append({
            "name": event.key,
            "count": event.count,
            "input_shapes": event.input_shapes,
            "self_cpu_time_us": event.self_cpu_time_total,
            "self_device_time_us": event.self_device_time_total,
            "device_time_us": event.device_time_total,
            "self_cpu_memory_bytes": event.self_cpu_memory_usage,
            "self_device_memory_bytes": event.self_device_memory_usage,
        })
    sort_key = "self_device_time_us" if device.type == "cuda" else "self_cpu_time_us"
    return sorted(rows, key=lambda row: row[sort_key], reverse=True)[:20]


def main(args):
    config_path = resolve_path(args.config)
    config = load_benchmark_config(config_path)
    batch_size = args.batch_size if args.batch_size is not None else config.batch_size
    sequence_length = (
        args.sequence_length if args.sequence_length is not None else config.sequence_length
    )
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    if sequence_length > config.model.context_length:
        raise ValueError("sequence_length cannot exceed model context_length")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    output_path = resolve_path(args.output)
    trace_path = resolve_path(args.trace) if args.trace else None
    if trace_path is not None and output_path == trace_path:
        raise ValueError("output and trace paths must be different")
    for path in (output_path, trace_path):
        if path is not None and path.exists():
            raise FileExistsError(f"Result already exists: {path}")

    device_name = args.device or config.device
    dtype_name = args.dtype or config.dtype
    backend = args.attention_backend or config.model.attention_backend
    device = select_device(device_name)
    if device.type == "cuda" and dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support BF16")

    torch.manual_seed(config.seed)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    dtype = getattr(torch, dtype_name)
    model_config = replace(config.model, attention_backend=backend)
    block = Block(model_config).to(device=device, dtype=dtype)
    block.train()

    shape = (batch_size, sequence_length, model_config.d_model)
    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    grad_output = torch.randn(shape, device=device, dtype=dtype)
    backward = args.workload == "forward_backward"

    validation = validate(block, x, grad_output, backward)
    samples = measure(
        block, x, grad_output, backward, device, args.warmup, args.iterations,
    )

    operator_summary = []
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        operator_summary = profile_operations(
            block, x, grad_output, backward, device, trace_path,
        )

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.resolve()),
        "shape": {
            "B": batch_size,
            "T": sequence_length,
            "D": model_config.d_model,
            "H": model_config.num_heads,
        },
        "dtype": dtype_name,
        "attention_backend": backend,
        "workload": args.workload,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "samples_seconds": samples,
        "mean_ms": statistics.mean(samples) * 1000,
        "median_ms": statistics.median(samples) * 1000,
        "parameter_count": sum(parameter.numel() for parameter in block.parameters()),
        **validation,
        "environment": {
            "device": str(device),
            "pytorch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "operator_summary": operator_summary,
        "trace_path": str(trace_path) if trace_path is not None else None,
        "notes": (
            "Timing and profiling are separate. Inputs and output gradients are preallocated; "
            "gradient reset is outside timing. Profiler data is not used as latency."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")

    print(
        f"block {args.workload}: mean={result['mean_ms']:.3f} ms | "
        f"median={result['median_ms']:.3f} ms"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main(parse_args())
