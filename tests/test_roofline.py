import json
import runpy

benchmark = runpy.run_path("scripts/benchmarks/roofline.py")
main = benchmark["main"]
parse_args = benchmark["parse_args"]


def test_cpu_smoke(tmp_path):
    output = tmp_path / "roofline.json"
    args = parse_args([
        "--device", "cpu",
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
    assert all(len(row["samples_seconds"]) == 2 for row in result["measurements"])
