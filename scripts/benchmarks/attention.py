"""Attention core only: preallocated Q/K/V, no projections, loss or optimizer."""

import argparse
import json
import math
import statistics
import time
import tomllib
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from llm_systems_lab.device import select_device, synchronize

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "batch_size": 2, "sequence_length": 512, "num_heads": 4, "head_dim": 32,
    "device": "mps", "dtype": "float32", "seed": 2137, "warmup": 20, "iterations": 1000,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Attention-core TOML, relative to the repo root.")
    for name in ("batch_size", "sequence_length", "num_heads", "head_dim", "seed", "warmup", "iterations"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"])
    parser.add_argument("--backend", choices=["naive", "sdpa"], required=True)
    parser.add_argument("--sdpa-kernel", choices=["auto", "math", "flash"], default="auto",
                        help="Force an SDPA implementation on CUDA only; no fallback.")
    parser.add_argument("--workload", choices=["forward", "forward_backward"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    settings = DEFAULTS.copy()
    if args.config:
        with (ROOT / args.config).open("rb") as file:
            config = tomllib.load(file)
        sections = {
            "attention": {"batch_size", "sequence_length", "num_heads", "head_dim"},
            "benchmark": {"device", "dtype", "seed", "warmup", "iterations"},
        }
        for section, values in config.items():
            if section not in sections or set(values) - sections[section]:
                raise ValueError(f"Unknown attention-core config fields in [{section}]")
            settings.update(values)
    # Explicit CLI flags win over TOML, then the original defaults.
    for name, value in settings.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    for name in ("batch_size", "sequence_length", "num_heads", "head_dim", "iterations", "warmup", "seed"):
        value = getattr(args, name)
        minimum = 0 if name in ("warmup", "seed") else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if args.dtype not in ("float32", "bfloat16"):
        raise ValueError("dtype must be float32 or bfloat16")
    if args.sdpa_kernel != "auto" and (
        args.backend != "sdpa" or torch.device(args.device).type != "cuda"
    ):
        raise ValueError("Explicit --sdpa-kernel requires --backend sdpa and --device cuda")
    return args


def attention(q, k, v, backend, mask):
    if backend == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(mask, float("-inf"))
    return scores.softmax(dim=-1) @ v


def check_correctness(q, k, v, grad_output, mask):
    # FP32 reference uses the SAME rounded inputs, not unrelated FP32 random values.
    inputs = [tensor.detach().float().requires_grad_() for tensor in (q, k, v)]
    reference = attention(*inputs, "naive", mask)
    reference_grads = torch.autograd.grad(reference, inputs, grad_output.float())
    rtol, atol = (5e-2, 2e-2) if q.dtype == torch.bfloat16 else (1e-4, 1e-5)
    errors = {}
    for backend in ("naive", "sdpa"):
        output = attention(q, k, v, backend, mask)
        gradients = torch.autograd.grad(output, (q, k, v), grad_output)
        actuals = [output, *gradients]
        expected = [reference, *reference_grads]
        # With no_grad the backend may select a different kernel.
        with torch.no_grad():
            actuals.append(attention(q, k, v, backend, mask))
        expected.append(reference)
        for name, actual, target in zip(("output", "dq", "dk", "dv", "no_grad_output"), actuals, expected):
            torch.testing.assert_close(actual.float(), target, rtol=rtol, atol=atol)
            errors[f"{backend}_{name}"] = (actual.float() - target).abs().max().item()
    return {"reference": "Naive FP32 on the same rounded inputs", "rtol": rtol, "atol": atol,
            "max_abs_errors": errors, "passed": True}


def cuda_peak(q, k, v, grad_output, mask, args, device):
    """Three separate warmed-up passes; no timing or saved-tensor hooks."""
    if device.type != "cuda":
        return None
    samples = []
    for _ in range(3):
        q.grad = k.grad = v.grad = None
        synchronize(device)
        before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.set_grad_enabled(args.workload == "forward_backward"):
            output = attention(q, k, v, args.backend, mask)
            if args.workload == "forward_backward":
                output.backward(grad_output)
        synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        samples.append({"before_bytes": before, "peak_bytes": peak, "extra_bytes": peak - before})
        del output
    q.grad = k.grad = v.grad = None
    return {"samples": samples, "note": (
        "After timing warmup/iterations. Peak PyTorch allocated bytes, not reserved or total GPU memory. "
        "Before includes inputs, upstream gradient and naive mask. Extra includes output, "
        "temporaries and input gradients (when backward runs). No empty_cache calls."
    )}


def saved_storage(q, k, v, backend, mask):
    """Untimed forward: count each saved allocation once, including its views."""
    preallocated = {tensor.untyped_storage().data_ptr() for tensor in (q, k, v, mask)}
    storages = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        key = storage.data_ptr()
        if key not in storages:
            storages[key] = {
                "bytes": storage.nbytes(),
                "preallocated": key in preallocated,
                "dtype": str(tensor.dtype),
                "shapes": [],
            }
        shape = list(tensor.shape)
        if shape not in storages[key]["shapes"]:
            storages[key]["shapes"].append(shape)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = attention(q, k, v, backend, mask)
    # Only metadata is kept; neither the output nor its graph is needed now.
    del output
    return {
        "total_bytes": sum(row["bytes"] for row in storages.values()),
        "extra_bytes": sum(row["bytes"] for row in storages.values() if not row["preallocated"]),
        "storages": list(storages.values()),
    }


def run_benchmark(args, device):
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Result already exists: {output_path}")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not support BF16")
    dtype = getattr(torch, args.dtype)
    B, T, H, head_dim = args.batch_size, args.sequence_length, args.num_heads, args.head_dim
    # Like GPT's head layout, but with independent input storages and leaf gradients.
    q, k, v = [
        torch.randn(B, T, H, head_dim).transpose(1, 2).to(device=device, dtype=dtype).requires_grad_()
        for _ in range(3)
    ]
    grad_output = torch.randn(B, H, T, head_dim, device=device).to(dtype)
    mask = torch.ones(T, T, dtype=torch.bool, device=device).triu(1)
    correctness = check_correctness(q, k, v, grad_output, mask)

    # SDPA does not use an explicit mask; release the one used for validation.
    if args.backend == "sdpa":
        del mask
        mask = torch.empty(0, dtype=torch.bool, device=device)

    backward = args.workload == "forward_backward"
    samples = []
    with torch.set_grad_enabled(backward):
        for index in range(args.warmup + args.iterations):
            q.grad = k.grad = v.grad = None
            synchronize(device)
            start = time.perf_counter()
            output = attention(q, k, v, args.backend, mask)
            if backward:
                output.backward(grad_output)
            synchronize(device)
            elapsed = time.perf_counter() - start
            # Do not keep the previous output/graph alive during the next iteration.
            # This cleanup and gradient reset are outside the timing.
            del output
            if index >= args.warmup:
                samples.append(elapsed)

    q.grad = k.grad = v.grad = None
    peak = cuda_peak(q, k, v, grad_output, mask, args, device)
    saved = saved_storage(q, k, v, args.backend, mask) if backward else {
        "total_bytes": 0, "extra_bytes": 0, "storages": [],
    }
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pytorch": torch.__version__,
        "device": str(device),
        "dtype": args.dtype,
        "config_path": str((ROOT / args.config).resolve()) if args.config else None,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else None,
        "cuda_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction if device.type == "cuda" else None
        ),
        "sdpa_math_reduced_precision_reduction": (
            torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed() if device.type == "cuda" else None
        ),
        "B": B, "T": T, "H": H, "head_dim": head_dim,
        "q_stride": list(q.stride()),
        "seed": args.seed,
        "backend": args.backend,
        "sdpa_kernel": args.sdpa_kernel if args.backend == "sdpa" else None,
        "sdpa_kernel_note": "auto is dispatcher-selected, not a claim that FlashAttention ran",
        "workload": args.workload,
        "description": (
            "Causal attention + backward with preallocated output gradient."
            if backward else "Causal attention under no_grad."
        ),
        "notes": "No projections/loss/optimizer. Inputs, mask and gradient reset outside timing.",
        "correctness": correctness,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "samples_seconds": samples,
        "mean_ms": statistics.mean(samples) * 1000,
        "median_ms": statistics.median(samples) * 1000,
        "saved_for_backward": saved,
        "cuda_peak": peak,
        "memory_note": (
            "Separate untimed forward; unique saved storages, NOT peak memory. "
            "Extra excludes preallocated Q/K/V and mask; views counted once. "
            "No_grad saves nothing, but still uses temporary memory."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(f"{args.backend} T={T} {args.workload}: "
          f"mean={result['mean_ms']:.3f} ms, median={result['median_ms']:.3f} ms, "
          f"saved extra={saved['extra_bytes'] / 2**20:.2f} MiB")
    print(f"Saved: {output_path}")


def main(args):
    device = select_device(args.device)
    kernels = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION}
    # Outside the timed loop. Only the requested CUDA kernel is enabled.
    context = sdpa_kernel(kernels[args.sdpa_kernel]) if args.sdpa_kernel != "auto" else nullcontext()
    with context:
        run_benchmark(args, device)


if __name__ == "__main__":
    main(parse_args())
