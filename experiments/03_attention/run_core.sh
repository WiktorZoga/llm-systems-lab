#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

for repeat in repeat-01 repeat-02 repeat-03; do
    # Alternate backend order to avoid always measuring SDPA second.
    backends="naive sdpa"
    if [ "$repeat" = "repeat-02" ]; then
        backends="sdpa naive"
    fi
    for length in 512 1024 2048; do
        for workload in forward forward_backward; do
            for backend in $backends; do
                uv run --locked python scripts/benchmarks/attention.py \
                    --sequence-length "$length" --backend "$backend" \
                    --workload "$workload" \
                    --output "experiments/03_attention/results/core/$repeat/$workload/$backend/t$length.json"
            done
        done
    done
done
