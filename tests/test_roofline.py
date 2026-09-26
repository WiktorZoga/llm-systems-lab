import json
import runpy

import pytest

benchmark = runpy.run_path("scripts/benchmarks/roofline.py")
main = benchmark["main"]
parse_args = benchmark["parse_args"]


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_cpu_smoke(tmp_path, dtype):
    output = tmp_path / "roofline.json"
    args = parse_args([
        "--device", "cpu",
        "--dtype", dtype,
        "--gemm-sizes", "16",
        "--copy-mib", "1",
        "--warmup", "1",
        "--iterations", "2",
        "--output", str(output),
    ])

    main(args)

    result = json.loads(output.read_text())
    assert result["gpu_name"] is None
    assert len(result["measurements"]) == 2
    assert result["measurements"][0]["operation"] == "gemm"
    assert result["measurements"][1]["operation"] == "copy"
    assert all(row["dtype"] == dtype for row in result["measurements"])
    assert result["measurements"][1]["bytes_moved"] == 2 * 2**20
    assert all(len(row["samples_seconds"]) == 2 for row in result["measurements"])
    assert dtype in result["notes"]
