#!/bin/bash

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG_DIR="configs/scaling"
RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_DIR="artifacts/benchmarks/$RUN_ID"
mkdir -p "$OUTPUT_DIR"

echo "Running all configs/scaling/ benchmarks"
echo "Saving results to: $OUTPUT_DIR"

for file in "$CONFIG_DIR"/*.toml; do
    if [ -f "$file" ]; then
        echo "Benchmarking: $file"
        uv run python scripts/benchmarks/benchmark.py \
            --config "$file" \
            --output-dir "$OUTPUT_DIR"
        sleep 5
    fi
done

echo "Finished all"
echo "Run directory: $OUTPUT_DIR"
