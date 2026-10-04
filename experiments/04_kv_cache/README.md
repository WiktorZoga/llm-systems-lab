# KV cache: prefill and decode

How much faster is decoding when we keep K/V from earlier tokens?

The no-cache benchmark works. `GPT.forward_cached` is the next implementation.
It will return last-token logits and a list of `(K, V)` pairs, one per layer,
with shape `[B, H, cached_length, head_dim]`. Positions continue from the cached
prefix; during one-token decode, the query can see every stored key.

## Run

From the repo root:

```bash
uv run python scripts/benchmarks/inference.py \
    --config configs/benchmarks/inference/baseline.toml \
    --output experiments/04_kv_cache/results/repeat-01/baseline_no_cache.json
```

`baseline` uses a 64-token prompt. `p32` and `p128` change only its length.
Each run has 3 warmup sequences and 10 measured sequences, with 32 decode calls.
Use a new filename for each run. Once cache works, add `--mode kv_cache` and
save as `baseline_kv_cache.json`; the script compares logits before timing.

Both modes use the same synthetic continuation and random weights.
Prefill processes the prompt; decode consumes one supplied token per call.
Tokens/s counts the whole batch and excludes prefill. Cache MiB is the final
size of K/V tensors. It does not include weights or temporary allocations.

## Plot

```bash
uv run python scripts/benchmarks/plot_inference.py \
    experiments/04_kv_cache/results/repeat-*/baseline_*.json \
    --output-dir experiments/04_kv_cache/figures
```

Compare the same model, prompt length, batch size and device.
This saves `latency.png`, `throughput.png` and `summary.csv`.
The notebook is there for exploring the individual measurements.
After collecting results, show a figure here with `![Latency](figures/latency.png)`.
