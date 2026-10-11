"""Compare full-prefix and cached inference on the same fixed token IDs."""

import argparse
import json
import statistics
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

import torch

from llm_systems_lab.config import ModelConfig
from llm_systems_lab.device import select_device, synchronize
from llm_systems_lab.models.gpt import GPT

ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmarks/inference/baseline.toml")
    parser.add_argument("--mode", choices=["no_cache", "kv_cache"], default="no_cache")
    parser.add_argument("--device")
    for name in ("batch-size", "prompt-length", "decode-steps", "warmup", "iterations"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--output", required=True, help="A new JSON file, relative to the repo root.")
    return parser.parse_args(argv)


def load_settings(args):
    with (ROOT / args.config).open("rb") as file:
        config = tomllib.load(file)
    model_config = ModelConfig(**config["model"])
    settings = config["benchmark"]
    for name in ("device", "batch_size", "prompt_length", "decode_steps", "warmup", "iterations"):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value

    for name in ("batch_size", "prompt_length", "decode_steps", "iterations"):
        if settings[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if settings["warmup"] < 0:
        raise ValueError("warmup must be non-negative")
    if settings["prompt_length"] + settings["decode_steps"] > model_config.context_length:
        raise ValueError("prompt_length + decode_steps must fit within context_length")
    if settings["dtype"] != "float32" or model_config.dropout != 0:
        raise ValueError("This first inference experiment uses float32 and dropout=0")
    return model_config, settings


def check_cached_logits(model, tokens, prompt_length):
    """Check each cached result against the same full prefix, outside timing."""
    cache = None
    max_error = 0.0
    for length in range(prompt_length, tokens.shape[1] + 1):
        prefix = tokens[:, :length].contiguous()
        expected, _ = model(prefix)
        new_tokens = prefix if cache is None else prefix[:, -1:].contiguous()
        actual, cache = model.forward_cached(new_tokens, cache)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

        shape = (tokens.shape[0], model.config.num_heads, length, model.config.head_dim)
        if len(cache) != model.config.num_layers:
            raise ValueError("Return one (K, V) pair per Transformer layer")
        for key, value in cache:
            if key.shape != shape or value.shape != shape:
                raise ValueError(f"Expected K and V of shape {shape}")
        max_error = max(max_error, (actual - expected).abs().max().item())
    # A separate greedy continuation starts with a fresh cache, outside timing.
    full_tokens = tokens[:, :prompt_length].clone()
    cached_tokens = full_tokens.clone()
    cache = None
    for _ in range(tokens.shape[1] - prompt_length):
        expected, _ = model(full_tokens)
        new_tokens = cached_tokens if cache is None else cached_tokens[:, -1:]
        actual, cache = model.forward_cached(new_tokens, cache)
        full_next = expected[:, -1].argmax(dim=-1, keepdim=True)
        cached_next = actual[:, -1].argmax(dim=-1, keepdim=True)
        if not torch.equal(full_next, cached_next):
            raise ValueError("Cached and full-prefix greedy tokens disagree")
        full_tokens = torch.cat((full_tokens, full_next), dim=1)
        cached_tokens = torch.cat((cached_tokens, cached_next), dim=1)
    return {
        "passed": True, "max_abs_error": max_error, "rtol": 1e-4, "atol": 1e-5,
        "greedy_tokens_match": True, "greedy_steps": tokens.shape[1] - prompt_length,
    }


def measure_sequence(model, inputs, device, use_cache):
    cache = None
    samples = []
    for input_ids in inputs:
        synchronize(device)
        start = time.perf_counter()
        if use_cache:
            logits, cache = model.forward_cached(input_ids, cache)
        else:
            logits, _ = model(input_ids)
        synchronize(device)
        samples.append(time.perf_counter() - start)
        # Do not keep the previous logits alive during the next forward.
        del logits

    # Logical K/V tensor sizes, not the allocator's peak or reserved memory.
    cache_bytes = 0
    if cache is not None:
        cache_bytes = sum(t.numel() * t.element_size() for pair in cache for t in pair)
    return samples[0], samples[1:], cache_bytes


def summarize(samples):
    return {
        "mean_ms": statistics.mean(samples) * 1000,
        "median_ms": statistics.median(samples) * 1000,
    }


def main(args):
    model_config, settings = load_settings(args)
    output_path = ROOT / args.output
    if output_path.exists():
        raise FileExistsError(f"Result already exists: {output_path}")

    device = select_device(settings["device"])
    torch.manual_seed(settings["seed"])
    torch.set_float32_matmul_precision("highest")
    model = GPT(model_config).to(device=device, dtype=torch.float32).eval()
    B, P, S = settings["batch_size"], settings["prompt_length"], settings["decode_steps"]
    tokens = torch.randint(model_config.vocab_size, (B, P + S)).to(device)

    use_cache = args.mode == "kv_cache"
    # Prepare slices outside timing. Both modes see exactly the same token sequence.
    inputs = [tokens[:, :P].contiguous()]
    for step in range(S):
        end = P + step + 1
        start = end - 1 if use_cache else 0
        inputs.append(tokens[:, start:end].contiguous())

    prefill_samples, decode_samples = [], []
    with torch.inference_mode():
        correctness = check_cached_logits(model, tokens, P) if use_cache else None
        for iteration in range(settings["warmup"] + settings["iterations"]):
            prefill, decode, cache_bytes = measure_sequence(model, inputs, device, use_cache)
            if iteration >= settings["warmup"]:
                prefill_samples.append(prefill)
                decode_samples.append(decode)

    flat_decode = [sample for sequence in decode_samples for sample in sequence]
    decode_totals = [sum(sequence) for sequence in decode_samples]
    result = {
        "mode": args.mode,
        "config": {"model": asdict(model_config), "benchmark": settings},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "environment": {"device": str(device), "pytorch": torch.__version__},
        "correctness": correctness,
        "prefill": {"samples_seconds": prefill_samples, **summarize(prefill_samples)},
        "decode": {
            "samples_seconds": decode_samples,
            "context_lengths": list(range(P + 1, P + S + 1)),
            **summarize(flat_decode),
            "tokens_per_second": B * S / statistics.mean(decode_totals),
        },
        "kv_cache_bytes": cache_bytes,
        "notes": "Synthetic tokens, random weights, inference mode. Timing includes synchronization. "
                 "Decode tokens/s excludes prefill. Cache bytes are final K/V storage, not peak memory.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(f"Prefill: mean={result['prefill']['mean_ms']:.3f} ms")
    print(f"Decode: mean={result['decode']['mean_ms']:.3f} ms/step "
          f"| {result['decode']['tokens_per_second']:.1f} tokens/s across batch")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main(parse_args())
