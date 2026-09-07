# Scaling configs

All configs use the same baseline: B=2, T=128, D=128, L=2, H=4,
V=50257, maximum context=256, dropout=0, naive attention, FP32, MPS.
Seed=2137, 5 warmup iterations, 100 measured iterations, AdamW lr=0.0003.

| Sweep | Config values (including baseline) | Only changed field |
|---|---|---|
| B | 1, 2, 4 | benchmark.batch_size |
| T | 64, 128, 256 | benchmark.sequence_length |
| D | 64, 128, 256, 512, 768 | model.d_model |
| L | 1, 2, 4, 8, 12, 16 | model.num_layers |
| H | 1, 2, 4, 8 | model.num_heads |
| V | 8192, 16384, 32768, 50257 | model.vocab_size |

Use baseline.toml for any baseline point. h4.toml, l2.toml and
v50257.toml are explicit copies of that point for convenient manual runs.
Changing H keeps D=128, so head_dim=D/H changes with H.
The GPT-2 124M config in the parent directory is a separate reference workload.

Older h*.json results used D=256. They are a separate head sweep at D=256;
editing the configs does not change those measurements.
The benchmark now releases outputs between iterations and explicitly selects
AdamW foreach=False/fused=False. Older JSONs claimed these optimizer settings
without enforcing them. Keep old and new measurements in separate directories.

## Timing contract

- forward: full-sequence logits + cross-entropy with autograd enabled.
- forward_backward: the same forward + backward; zero_grad outside timing.
- optimizer_step: zero_grad(set_to_none=True), forward, backward, AdamW.step.

Wall-clock timing includes Python dispatch and device completion. Synchronize
before and after each timed iteration. Outputs are released outside timing.
Each workload starts with the same seeded weights and synthetic batch;
optimizer steps update weights, including during warmup. At least one warmup
step is needed to exclude lazy AdamW state initialization from measured steps.
Allocator caches are left warm. Tokens/s = B*T / mean_seconds for each workload.
Store all samples, mean and median; neither statistic proves run-to-run stability.

## Commands (run manually)

Run from the repository root, using a fresh result directory for each repeat:

```bash
uv run python scripts/benchmarks/benchmark.py --config configs/benchmarks/scaling/baseline.toml --output-dir artifacts/benchmarks/scaling-repeat-01
uv run python scripts/benchmarks/summarize_benchmarks.py --input-dir artifacts/benchmarks/scaling-repeat-01
```

For a complete sweep use `bash scripts/benchmarks/run_all.sh`. It prints its
dated output directory; pass that directory to `plot_all.sh`.
For an individual plot pass the JSON files, not the TOML configs:

```bash
uv run python scripts/benchmarks/plot_benchmarks.py --parameter num_layers --statistic median --output-dir artifacts/benchmarks/manual/plots artifacts/benchmarks/manual/l1.json artifacts/benchmarks/manual/l2.json artifacts/benchmarks/manual/l4.json
```

## Memory snapshots

```bash
uv run python scripts/benchmarks/memory.py --config configs/benchmarks/scaling/baseline.toml --device mps --output artifacts/memory/baseline-01.json
```

Run in a fresh process. Six synchronized snapshots cover before/after model
creation, after optimizer creation, forward, backward and the first optimizer
step. The batch is allocated after optimizer creation. Logits, loss and gradients
are retained through the final snapshot. No warmup or empty_cache calls.

MPS tensor_bytes excludes allocator caches; driver_bytes includes caches and
Metal framework allocations. These are device-side snapshots, not total system
RAM and not an MPS peak. CUDA additionally records reserved_bytes and cumulative
peak_tensor_bytes since the reset before model creation. CPU reports null.
Do not compare MPS driver_bytes directly with CUDA reserved_bytes.

## SDPA A/B (not yet measured)

Before measuring, compare logits, loss and gradients with identical weights
and inputs. Then run the same config in separate processes/directories:

```bash
uv run python scripts/benchmarks/benchmark.py --config configs/benchmarks/scaling/baseline.toml --attention-backend naive --output-dir artifacts/benchmarks/attention-naive-01
uv run python scripts/benchmarks/benchmark.py --config configs/benchmarks/scaling/baseline.toml --attention-backend sdpa --output-dir artifacts/benchmarks/attention-sdpa-01
```

Both use identical seeded weights, tokens and optimizer settings. Repeat runs
and alternate their order. Compare the same workload and statistic in the two
JSONs/CSV tables. The scaling plot deliberately rejects mixed attention backends.
SDPA is selected by PyTorch; using this API is not proof of a FlashAttention
kernel or a speedup. Only naive attention allocates the explicit causal mask.
It is recreated at model construction and is not stored in new checkpoints.
The generation script ignores saved masks when loading older checkpoints.

CUDA uses the same commands with `--device cuda:0`. FP32 matmul precision is
set to highest; JSON records the device and PyTorch version.
Record the hardware name yourself alongside the results you publish.
No CUDA performance result is claimed until actually measured.
