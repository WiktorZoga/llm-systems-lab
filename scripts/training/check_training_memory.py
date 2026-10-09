"""Check training equivalence and inspect what autograd saves for backward."""

import argparse
import json
import tomllib
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import AdamW

from llm_systems_lab.config import ModelConfig
from llm_systems_lab.device import select_device, synchronize
from llm_systems_lab.models.gpt import GPT

ROOT = Path(__file__).resolve().parents[2]
RTOL, ATOL = 1e-4, 1e-5


def run_step(
    model_config, settings, inputs, targets, device, *, checkpoint_blocks=False,
    micro_batch_size=None, inspect=False, dtype=torch.float32,
):
    # Every case starts from identical CPU-initialized weights.
    torch.manual_seed(settings["seed"])
    model = GPT(model_config).to(device=device, dtype=dtype).train()
    optimizer = AdamW(model.parameters(), lr=settings["learning_rate"], foreach=False, fused=False)
    micro_batch_size = micro_batch_size or settings["batch_size"]
    accumulation_steps = settings["batch_size"] // micro_batch_size

    # We want activation storage, not memory already occupied by the model or batch.
    existing = {
        tensor.untyped_storage().data_ptr()
        for tensor in [*model.parameters(), *model.buffers(), inputs, targets]
    }
    saved = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        # Views share storage. Count its bytes once, excluding existing weights/inputs.
        if storage.data_ptr() not in existing:
            saved[storage.data_ptr()] = storage.nbytes()
        return tensor

    calls = 0

    def count_forward(module, args):
        # Update the enclosing counter whenever a block starts, including recomputation.
        nonlocal calls
        calls += 1

    handles = [block.register_forward_pre_hook(count_forward) for block in model.transformer["h"]]
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0
    forward_calls, backward_calls = 0, 0
    for start in range(0, settings["batch_size"], micro_batch_size):
        end = start + micro_batch_size
        before_forward = calls
        # Inspect tensors when autograd saves them; return them unchanged for backward.
        context = (
            torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor)
            if inspect else nullcontext()
        )
        with context:
            logits, loss = model(
                inputs[start:end], targets[start:end], checkpoint_blocks=checkpoint_blocks,
            )
        scaled_loss = loss / accumulation_steps
        total_loss = total_loss + scaled_loss.detach()
        before_backward = calls
        forward_calls += calls - before_forward
        scaled_loss.backward()
        backward_calls += calls - before_backward
        # Drop this microbatch's references before the next one. This is not empty_cache().
        del logits, loss, scaled_loss

    synchronize(device)
    gradients = {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters()}
    optimizer.step()
    synchronize(device)
    parameters = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    for handle in handles:
        handle.remove()
    return {
        "loss": total_loss.item(), "gradients": gradients, "parameters": parameters,
        "saved_activation_bytes": sum(saved.values()) if inspect else None,
        "forward_block_calls": forward_calls, "backward_block_calls": backward_calls,
    }


def compare(actual, reference, *, check_parameters=True, rtol=RTOL, atol=ATOL):
    torch.testing.assert_close(actual["loss"], reference["loss"], rtol=rtol, atol=atol)
    errors = {
        "loss_abs_error": abs(actual["loss"] - reference["loss"]),
        "parameters_checked": check_parameters,
    }
    for field in ("gradients", "parameters"):
        maximum = 0.0
        for name, expected in reference[field].items():
            tensor = actual[field][name]
            if field == "gradients" or check_parameters:
                torch.testing.assert_close(
                    tensor, expected, rtol=rtol, atol=atol, msg=f"{field}: {name}",
                )
            maximum = max(maximum, (tensor - expected).abs().max().item())
        errors[f"{field}_max_abs_error"] = maximum
    return errors


def main(args):
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Result already exists: {output}")
    with (ROOT / args.config).open("rb") as file:
        raw = tomllib.load(file)
    model_config = ModelConfig(**raw["model"])
    settings = raw["train"]
    B, T, micro = settings["batch_size"], settings["sequence_length"], settings["micro_batch_size"]
    if B <= 0 or micro <= 0 or B % micro or T <= 0 or T > model_config.context_length:
        raise ValueError("Use positive B/T/microbatch, B divisible by microbatch, and T within context")
    if model_config.dropout != 0:
        raise ValueError("This deterministic comparison uses dropout=0")
    device = select_device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "mps" and dtype == torch.float64:
        raise ValueError("MPS does not support float64; use CPU for the high-precision check")
    rtol, atol = (1e-7, 1e-9) if dtype == torch.float64 else (RTOL, ATOL)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(settings["seed"])
    tokens = torch.randint(model_config.vocab_size, (B, T + 1))
    inputs = tokens[:, :-1].contiguous().to(device)
    targets = tokens[:, 1:].contiguous().to(device)

    reference = run_step(model_config, settings, inputs, targets, device, inspect=True, dtype=dtype)
    checkpointed = run_step(
        model_config, settings, inputs, targets, device,
        checkpoint_blocks=True, inspect=True, dtype=dtype,
    )
    checkpoint_errors = compare(checkpointed, reference, rtol=rtol, atol=atol)
    accumulated = run_step(model_config, settings, inputs, targets, device, micro_batch_size=micro, dtype=dtype)
    # FP32 rounding near zero gradients can be amplified by the first AdamW update.
    # Verify updates strictly in the float64 check; report the FP32 difference.
    accumulation_errors = compare(
        accumulated, reference, check_parameters=dtype == torch.float64, rtol=rtol, atol=atol,
    )
    # Check the combined path used by the real training config too.
    combined = run_step(
        model_config, settings, inputs, targets, device,
        micro_batch_size=micro, checkpoint_blocks=True, dtype=dtype,
    )
    combined_errors = compare(combined, accumulated, rtol=rtol, atol=atol)

    cases = []
    for name, result in [("normal", reference), ("checkpointed", checkpointed)]:
        cases.append({key: value for key, value in {"name": name, **result}.items()
                      if key not in ("gradients", "parameters")})
    result = {
        "model": asdict(model_config), "train": settings,
        "device": str(device), "pytorch": torch.__version__, "dtype": args.dtype,
        "optimizer": {"name": "AdamW", "lr": settings["learning_rate"], "foreach": False, "fused": False},
        "tolerances": {"rtol": rtol, "atol": atol},
        "correctness": {"checkpointing": checkpoint_errors, "accumulation": accumulation_errors,
                        "combined": combined_errors},
        "cases": cases,
        "notes": "Saved activation bytes count unique storages saved during one full-batch forward, "
                 "excluding pre-existing parameters, buffers and input/target tensors. This is not peak memory. "
                 "Block calls count entries, not FLOPs; non-reentrant recomputation may stop early. "
                 "All cases use identical weights/tokens and equal-sized microbatches. Combined compares "
                 "checkpointed microbatches against ordinary microbatches. FP32 accumulation update "
                 "errors are reported, not asserted; the float64 CPU run checks them strictly. No latency is measured.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    for case in cases:
        print(f"{case['name']}: saved={case['saved_activation_bytes'] / 2**20:.2f} MiB | "
              f"block calls forward/backward={case['forward_block_calls']}/{case['backward_block_calls']}")
    print(f"Loss/gradient and checkpointing update checks passed. Saved: {output}")
    if dtype == torch.float32:
        print(f"FP32 accumulation AdamW update max difference: {accumulation_errors['parameters_max_abs_error']:.3e} (reported, not asserted)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_memory.toml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
