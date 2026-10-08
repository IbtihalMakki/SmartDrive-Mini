
"""Compare SmartDrive RGB-D localization performance in Python and C++17."""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autonomy.metric_localization import (
    camera_intrinsics,
    localize_rgbd,
)


# ------------------------------------------------------------
# Benchmark utilities
# ------------------------------------------------------------

def percentile95(values):
    return float(np.percentile(values, 95))


# ------------------------------------------------------------
# Python benchmark
# ------------------------------------------------------------

def benchmark_python(repetitions=2000, warmup=200):
    width, height = 640, 480

    # Generate deterministic synthetic depth data.
    yy, xx = np.indices((height, width))

    depth = np.where(
        (xx + yy) % 17 == 0,
        0.0,
        8.0 + ((xx * 3 + yy * 7) % 101) * 0.01,
    ).astype(np.float64)

    intrinsics = camera_intrinsics(
        width,
        height,
        90.0,
    )

    objects = [
        {
            "track_id": 1,
            "bbox": [200, 120, 440, 360],
            "distance_valid": True,
            "depth_frame_id": 1,
            "distance_m": 10.0,
        }
    ]

    def run():
        return localize_rgbd(
            objects,
            depth,
            1,
            intrinsics,
        )[0]

    # Warmup iterations.
    for _ in range(warmup):
        run()

    samples = []
    result = None

    # Timed iterations.
    for _ in range(repetitions):
        start = time.perf_counter()

        result = run()

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        samples.append(elapsed_ms)

    if result is None or not result["localization_valid"]:
        raise RuntimeError(
            "Python RGB-D localization failed"
        )

    mean_ms = float(np.mean(samples))

    return {
        "language": "python_numpy",
        "iterations": repetitions,
        "warmup": warmup,
        "mean_ms": mean_ms,
        "p95_ms": percentile95(samples),
        "throughput_ops_s": 1000.0 / mean_ms,
        "position": result["position_3d_camera_m"],
        "sample_count": result["localization_sample_count"],
    }


# ------------------------------------------------------------
# C++ benchmark
# ------------------------------------------------------------

def benchmark_cpp(
    executable,
    repetitions=2000,
    warmup=200,
):
    completed = subprocess.run(
        [
            str(executable),
            str(repetitions),
            str(warmup),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(completed.stdout)


# ------------------------------------------------------------
# Main comparison
# ------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python compare_performance.py "
            "geometry_benchmark.exe"
        )

    executable = Path(sys.argv[1]).resolve()

    if not executable.is_file():
        raise FileNotFoundError(
            f"C++ executable not found: {executable}"
        )

    repetitions = 2000
    warmup = 200

    # Run C++ benchmark.
    cpp_result = benchmark_cpp(
        executable,
        repetitions,
        warmup,
    )

    # Run Python benchmark.
    python_result = benchmark_python(
        repetitions,
        warmup,
    )

    # --------------------------------------------------------
    # Numerical parity
    # --------------------------------------------------------

    cpp_position = np.asarray(
        cpp_result["position"],
        dtype=float,
    )

    python_position = np.asarray(
        python_result["position"],
        dtype=float,
    )

    position_error = float(
        np.linalg.norm(
            cpp_position - python_position
        )
    )

    sample_counts_match = (
        cpp_result["sample_count"]
        == python_result["sample_count"]
    )

    numerical_parity = bool(
        position_error < 1e-6
        and sample_counts_match
    )

    # --------------------------------------------------------
    # Performance comparison
    # --------------------------------------------------------

    speedup = (
        python_result["mean_ms"]
        / cpp_result["mean_ms"]
    )

    report = {
        "benchmark": (
            "SmartDrive RGB-D visible-surface localization"
        ),
        "image_size": [640, 480],
        "bbox": [200, 120, 440, 360],
        "cpp": cpp_result,
        "python": python_result,
        "comparison": {
            "cpp_speedup_vs_python": speedup,
            "position_error_m": position_error,
            "sample_counts_match": sample_counts_match,
            "numerical_parity": numerical_parity,
        },
        "limitations": [
            "Synthetic deterministic depth input",
            "Single bounding box",
            "Single-process C++ timing excludes process startup",
            "Python timing includes Python and NumPy function overhead",
            "No CARLA runtime or full-pipeline FPS measured",
        ],
    }

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output = (
        PROJECT_ROOT
        / "results"
        / "rgbd_cpp_python_performance.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("SMARTDRIVE RGB-D PERFORMANCE BENCHMARK")
    print("-" * 48)

    print(
        f"Python mean latency: "
        f"{python_result['mean_ms']:.4f} ms"
    )

    print(
        f"C++ mean latency:    "
        f"{cpp_result['mean_ms']:.4f} ms"
    )

    print(
        f"Python P95:         "
        f"{python_result['p95_ms']:.4f} ms"
    )

    print(
        f"C++ P95:            "
        f"{cpp_result['p95_ms']:.4f} ms"
    )

    print(
        f"Python throughput:  "
        f"{python_result['throughput_ops_s']:.2f} ops/s"
    )

    print(
        f"C++ throughput:     "
        f"{cpp_result['throughput_ops_s']:.2f} ops/s"
    )

    print(
        f"C++ speedup:        "
        f"{speedup:.2f}x"
    )

    print(
        f"Position error:     "
        f"{position_error:.10f} m"
    )

    print(
        f"Numerical parity:   "
        f"{numerical_parity}"
    )

    print()
    print(f"Saved report: {output}")

    if not numerical_parity:
        raise SystemExit(
            "Numerical parity check failed"
        )


if __name__ == "__main__":
    main()
