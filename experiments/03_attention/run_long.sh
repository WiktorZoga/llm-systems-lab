#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Full forward + loss. Only autograd and attention backend change within a pair.
for repeat in 01 02 03; do
  for mode in grad no_grad; do
    flags=(--workload forward)
    if [ "$mode" = no_grad ]; then
      flags+=(--forward-no-grad)
    fi
    for length in 256 512 1024; do
      for backend in naive sdpa; do
        echo "Time: repeat-$repeat / $mode / $backend / T=$length"
        uv run --locked python scripts/benchmarks/benchmark.py \
          --config "configs/benchmarks/attention/t$length.toml" \
          --attention-backend "$backend" "${flags[@]}" \
          --output-dir "experiments/03_attention/results/long/repeat-$repeat/$mode/$backend"
      done
    done
  done
done

# Memory is measured separately: one 1000-step run per combination.
for mode in grad no_grad; do
  flags=(--workload forward)
  if [ "$mode" = no_grad ]; then
    flags+=(--forward-no-grad)
  fi
  for length in 256 512 1024; do
    for backend in naive sdpa; do
      echo "Memory: $mode / $backend / T=$length"
      uv run --locked python scripts/benchmarks/memory.py \
        --config "configs/benchmarks/attention/t$length.toml" \
        --attention-backend "$backend" "${flags[@]}" \
        --output "experiments/03_attention/results/long/memory/$mode/$backend/t$length.json"
    done
  done
done
