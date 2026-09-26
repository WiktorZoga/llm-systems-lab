"""Small GEMM and tensor-copy measurements for a practical roofline reference."""

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from llm_systems_lab.device import select_device, synchronize

ROOT = Path(__file__).resolve().parents[2]


def measure(operation, device, warmup, iterations):
    samples = []
    for index in range(warmup + iterations):
        synchronize(device)
        start = time.perf_counter()
        operation()
        synchronize(device)
        elapsed = time.perf_counter() - start
        if index >= warmup:
            samples.append(elapsed)
    return samples


def summarize(samples, flops=0, bytes_moved=0):
    mean_seconds = statistics.mean(samples)
    return {
        "samples_seconds": samples,
        "mean_ms": mean_seconds * 1000,
        "median_ms": statistics.median(samples) * 1000,
        "flops": flops,
        "bytes_moved": bytes_moved,
        "gflops": flops / mean_seconds / 1e9 if flops else None,
        "gb_per_second": bytes_moved / mean_seconds / 1e9 if bytes_moved else None,
    }


def run_gemm(size, device, dtype, warmup, iterations):
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    output = torch.empty(size, size, device=device, dtype=dtype)
    samples = measure(lambda: torch.mm(a, b, out=output), device, warmup, iterations)
    return {
        "operation": "gemm",
        "size": size,
        "dtype": str(dtype).removeprefix("torch."),
        **summarize(samples, flops=2 * size**3, bytes_moved=3 * a.numel() * a.element_size()),
    }


def run_copy(size_mib, device, dtype, warmup, iterations):
    num_bytes = size_mib * 2**20
    element_size = torch.empty((), dtype=dtype).element_size()
    source = torch.randn(num_bytes // element_size, device=device, dtype=dtype)
    destination = torch.empty_like(source)
    samples = measure(lambda: destination.copy_(source), device, warmup, iterations)
    return {
        "operation": "copy",
        "size_mib": size_mib,
        "dtype": str(dtype).removeprefix("torch."),
        # A copy reads the source and writes the destination.
        **summarize(samples, bytes_moved=2 * num_bytes),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="cpu, mps, cuda or cuda:N")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--gemm-sizes", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--copy-mib", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", default="artifacts/benchmarks/roofline.json")
    return parser.parse_args(argv)


def main(args):
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if any(size <= 0 for size in [*args.gemm_sizes, *args.copy_mib]):
        raise ValueError("sizes must be positive")

    device = select_device(args.device)
    if device.type == "cuda" and args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support BF16")
    dtype = getattr(torch, args.dtype)
    measurements = []
    for size in args.gemm_sizes:
        measurements.append(run_gemm(size, device, dtype, args.warmup, args.iterations))
    for size_mib in args.copy_mib:
        measurements.append(run_copy(size_mib, device, dtype, args.warmup, args.iterations))

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "dtype": args.dtype,
        "pytorch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "notes": (
            f"GEMM uses preallocated {args.dtype} inputs and output. Copy reads one "
            f"preallocated {args.dtype} tensor and writes another. Synchronization is "
            "outside each timed operation."
        ),
        "measurements": measurements,
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError(f"Result already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")

    for row in measurements:
        label = f"GEMM {row['size']}" if row["operation"] == "gemm" else f"copy {row['size_mib']} MiB"
        print(f"{label}: median={row['median_ms']:.3f} ms")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main(parse_args())
