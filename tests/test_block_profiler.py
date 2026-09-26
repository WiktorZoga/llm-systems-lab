import json
import runpy
import subprocess
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
benchmark = runpy.run_path(str(ROOT / "scripts/benchmarks/profile_block.py"))
main = benchmark["main"]
parse_args = benchmark["parse_args"]


def test_cpu_forward_backward_smoke(tmp_path):
    output = tmp_path / "result.json"
    trace = tmp_path / "trace.json"
    args = parse_args([
        "--config", "configs/benchmarks/scaling/baseline.toml",
        "--device", "cpu",
        "--workload", "forward_backward",
        "--batch-size", "1",
        "--sequence-length", "8",
        "--warmup", "1",
        "--iterations", "2",
        "--output", str(output),
        "--trace", str(trace),
    ])

    main(args)

    result = json.loads(output.read_text())
    assert result["shape"] == {"B": 1, "T": 8, "D": 128, "H": 4}
    assert result["workload"] == "forward_backward"
    assert result["output_shape"] == [1, 8, 128]
    assert result["output_finite"]
    assert result["gradients_finite"]
    assert len(result["samples_seconds"]) == 2
    assert result["mean_ms"] > 0
    assert result["median_ms"] > 0
    assert result["operator_summary"]
    assert trace.is_file()

    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(args)
    assert output.read_bytes() == original


def test_existing_trace_is_not_overwritten(tmp_path):
    output = tmp_path / "result.json"
    trace = tmp_path / "trace.json"
    trace.write_text("keep me")
    args = parse_args([
        "--config", "configs/benchmarks/scaling/baseline.toml",
        "--device", "cpu",
        "--workload", "forward",
        "--batch-size", "1",
        "--sequence-length", "8",
        "--warmup", "0",
        "--iterations", "1",
        "--output", str(output),
        "--trace", str(trace),
    ])

    with pytest.raises(FileExistsError):
        main(args)
    assert not output.exists()
    assert trace.read_text() == "keep me"


def test_output_and_trace_must_be_different(tmp_path):
    path = tmp_path / "result.json"
    args = parse_args([
        "--config", "configs/benchmarks/scaling/baseline.toml",
        "--device", "cpu",
        "--workload", "forward",
        "--batch-size", "1",
        "--sequence-length", "8",
        "--warmup", "0",
        "--iterations", "1",
        "--output", str(path),
        "--trace", str(path),
    ])

    with pytest.raises(ValueError, match="different"):
        main(args)
    assert not path.exists()


def test_sequence_length_cannot_exceed_context(tmp_path):
    args = parse_args([
        "--config", "configs/benchmarks/scaling/baseline.toml",
        "--device", "cpu",
        "--workload", "forward",
        "--sequence-length", "512",
        "--output", str(tmp_path / "result.json"),
    ])

    with pytest.raises(ValueError, match="context_length"):
        main(args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA device")
def test_cuda_smoke(tmp_path):
    output = tmp_path / "result.json"
    trace = tmp_path / "trace.json"
    args = parse_args([
        "--config", "configs/benchmarks/scaling/baseline.toml",
        "--device", "cuda",
        "--workload", "forward_backward",
        "--batch-size", "1",
        "--sequence-length", "8",
        "--warmup", "1",
        "--iterations", "2",
        "--output", str(output),
        "--trace", str(trace),
    ])

    main(args)

    result = json.loads(output.read_text())
    assert result["environment"]["gpu_name"]
    assert any(row["self_device_time_us"] > 0 for row in result["operator_summary"])


def test_cuda_runner_rejects_existing_output_directory(tmp_path):
    output = tmp_path / "session"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep me")

    completed = subprocess.run(
        ["bash", str(ROOT / "scripts/benchmarks/run_cuda_block_profile.sh"), str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "already exists" in completed.stderr
    assert list(output.iterdir()) == [marker]
