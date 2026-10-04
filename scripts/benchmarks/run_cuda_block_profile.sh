#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

OUTPUT_DIR="${1:-artifacts/cuda-block-profile}"
if [ -e "$OUTPUT_DIR" ]; then
    echo "Output directory already exists: $OUTPUT_DIR" >&2
    exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
    echo "Commit or stash local changes before the CUDA session." >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR/roofline" "$OUTPUT_DIR/timings" "$OUTPUT_DIR/traces"

git rev-parse HEAD > "$OUTPUT_DIR/commit.txt"
nvidia-smi > "$OUTPUT_DIR/nvidia-smi.txt"
uv run --locked python -c \
    'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.version.cuda); print("GPU:", torch.cuda.get_device_name())' \
    > "$OUTPUT_DIR/environment.txt"

for dtype in float32 bfloat16; do
    for repeat in 01 02 03; do
        uv run --locked python scripts/benchmarks/roofline.py \
            --device cuda \
            --dtype "$dtype" \
            --gemm-sizes 2048 4096 8192 \
            --copy-mib 64 256 1024 \
            --warmup 10 \
            --iterations 50 \
            --output "$OUTPUT_DIR/roofline/${dtype}-repeat-${repeat}.json"
    done
done

run_block() {
    name="$1"
    config="$2"
    workload="$3"
    shift 3

    for repeat in 01 02 03; do
        trace_args=()
        if [ "$repeat" = "01" ]; then
            trace_args=(--trace "$OUTPUT_DIR/traces/${name}-${workload}.json")
        fi

        uv run --locked python scripts/benchmarks/profile_block.py \
            --config "$config" \
            --device cuda \
            --dtype float32 \
            --attention-backend naive \
            --workload "$workload" \
            --warmup 10 \
            --iterations 50 \
            --output "$OUTPUT_DIR/timings/${name}-${workload}-repeat-${repeat}.json" \
            "${trace_args[@]}" \
            "$@"
    done
}

run_block tiny configs/benchmarks/scaling/baseline.toml forward
run_block tiny configs/benchmarks/scaling/baseline.toml forward_backward
run_block gpt2-block configs/benchmarks/cuda/gpt2_124m.toml forward \
    --batch-size 2 --sequence-length 1024
run_block gpt2-block configs/benchmarks/cuda/gpt2_124m.toml forward_backward \
    --batch-size 2 --sequence-length 1024

echo "Saved CUDA session outputs to $OUTPUT_DIR"
