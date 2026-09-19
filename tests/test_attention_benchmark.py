import json
import runpy
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
benchmark = runpy.run_path(str(ROOT / "scripts/benchmarks/attention.py"))
parse_args = benchmark["parse_args"]
main = benchmark["main"]
REQUIRED = ["--backend", "sdpa", "--workload", "forward", "--output", "unused.json"]


def test_defaults_and_cli_overrides():
    args = parse_args(REQUIRED)
    assert (args.batch_size, args.num_heads, args.head_dim, args.sequence_length) == (2, 4, 32, 512)
    assert (args.device, args.dtype, args.warmup, args.iterations) == ("mps", "float32", 20, 1000)
    args = parse_args(REQUIRED + [
        "--config", "configs/benchmarks/attention_core/tiny_t1024.toml",
        "--device", "cpu", "--dtype", "float32", "--warmup", "0", "--iterations", "2",
    ])
    assert args.sequence_length == 1024
    assert (args.device, args.dtype, args.warmup, args.iterations) == ("cpu", "float32", 0, 2)


def test_config_shapes():
    expected = {"tiny_t512": (4, 32, 512), "tiny_t1024": (4, 32, 1024),
                "tiny_t2048": (4, 32, 2048), "gpt2_heads_t1024": (12, 64, 1024)}
    for name, shape in expected.items():
        args = parse_args(REQUIRED + ["--config", f"configs/benchmarks/attention_core/{name}.toml"])
        assert (args.num_heads, args.head_dim, args.sequence_length) == shape
        assert (args.batch_size, args.seed, args.warmup, args.iterations) == (2, 2137, 20, 1000)
        assert (args.device, args.dtype) == ("cuda", "bfloat16")


@pytest.mark.parametrize("flags", [
    ["--device", "mps", "--sdpa-kernel", "flash"],
    ["--device", "cpu", "--sdpa-kernel", "math"],
    ["--device", "cuda", "--backend", "naive", "--sdpa-kernel", "flash"],
    ["--iterations", "0"],
])
def test_invalid_options(flags):
    with pytest.raises(ValueError):
        parse_args(REQUIRED + flags)


def test_config_typo_is_not_ignored(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text('[attention]\nhead_dims = 32\n')
    with pytest.raises(ValueError, match="Unknown"):
        parse_args(REQUIRED + ["--config", str(path)])


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("backend,workload", [("naive", "forward"), ("sdpa", "forward_backward")])
def test_cpu_smoke(tmp_path, dtype, backend, workload):
    path = tmp_path / "smoke.json"
    args = parse_args([
        "--device", "cpu", "--dtype", dtype, "--backend", backend, "--workload", workload,
        "--batch-size", "1", "--num-heads", "2", "--head-dim", "8", "--sequence-length", "8",
        "--warmup", "1", "--iterations", "2", "--output", str(path),
    ])
    main(args)
    result = json.loads(path.read_text())
    assert result["correctness"]["passed"]
    assert result["dtype"] == dtype
    assert result["cuda_peak"] is None
    assert result["gpu_name"] is None
    assert len(result["samples_seconds"]) == 2
    assert all(sample > 0 for sample in result["samples_seconds"])
    assert result["saved_for_backward"]["total_bytes"] == sum(
        s["bytes"] for s in result["saved_for_backward"]["storages"]
    )
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        main(args)
    assert path.read_bytes() == original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA device")
@pytest.mark.parametrize("workload", ["forward", "forward_backward"])
@pytest.mark.parametrize("kernel,dtype", [("math", "float32"), ("math", "bfloat16"), ("flash", "bfloat16")])
def test_cuda_peak(tmp_path, workload, kernel, dtype):
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        pytest.skip("Requires BF16 support")
    if kernel == "flash" and not torch.backends.cuda.is_flash_attention_available():
        pytest.skip("Requires a CUDA build with FlashAttention")
    path = tmp_path / "cuda.json"
    args = parse_args([
        "--device", "cuda", "--backend", "sdpa", "--sdpa-kernel", kernel, "--dtype", dtype,
        "--workload", workload, "--sequence-length", "32",
        "--warmup", "1", "--iterations", "2", "--output", str(path),
    ])
    main(args)
    result = json.loads(path.read_text())
    assert result["gpu_name"]
    assert result["sdpa_kernel"] == kernel
    samples = result["cuda_peak"]["samples"]
    assert len(samples) == 3
    for sample in samples:
        assert sample["peak_bytes"] > sample["before_bytes"]
        assert sample["extra_bytes"] == sample["peak_bytes"] - sample["before_bytes"]
