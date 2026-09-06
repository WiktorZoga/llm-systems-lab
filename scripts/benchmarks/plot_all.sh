#!/bin/bash

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [ "$#" -ne 1 ]; then
    echo "Usage: bash scripts/benchmarks/plot_all.sh RUN_DIRECTORY"
    exit 1
fi

RUN_DIR="$1"

if [[ "$RUN_DIR" != /* ]]; then
    RUN_DIR="$ROOT/$RUN_DIR"
fi

if [ ! -d "$RUN_DIR" ]; then
    echo "Run directory not found: $RUN_DIR"
    exit 1
fi

PLOT_SCRIPT="scripts/benchmarks/plot_benchmarks.py"
PLOT_DIR="$RUN_DIR/plots"

plot_sweep() {
    parameter="$1"
    shift

    echo "Plotting: $parameter"

    uv run python "$PLOT_SCRIPT" \
        --parameter "$parameter" \
        --output-dir "$PLOT_DIR" \
        "$@"
}

plot_sweep batch_size \
    "$RUN_DIR/b1.json" \
    "$RUN_DIR/baseline.json" \
    "$RUN_DIR/b4.json"

plot_sweep sequence_length \
    "$RUN_DIR/t64.json" \
    "$RUN_DIR/baseline.json" \
    "$RUN_DIR/t256.json"

plot_sweep d_model \
    "$RUN_DIR/d64.json" \
    "$RUN_DIR/baseline.json" \
    "$RUN_DIR/d256.json" \
    "$RUN_DIR/d512.json" \
    "$RUN_DIR/d768.json"

plot_sweep num_layers \
    "$RUN_DIR/l1.json" \
    "$RUN_DIR/baseline.json" \
    "$RUN_DIR/l4.json"

plot_sweep vocab_size \
    "$RUN_DIR/v8192.json" \
    "$RUN_DIR/v16384.json" \
    "$RUN_DIR/v32768.json" \
    "$RUN_DIR/v50257.json"

plot_sweep num_heads \
    "$RUN_DIR/h1.json" \
    "$RUN_DIR/h2.json" \
    "$RUN_DIR/h4.json" \
    "$RUN_DIR/h8.json"

echo "Finished plotting"
