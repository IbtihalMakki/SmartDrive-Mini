# SmartDrive-Mini

Incremental CARLA autonomous-perception prototype. Existing RGB-D distance,
traffic-light and recovery validations are retained. This document currently
records validated Phases 1–2 and Phase 3 evaluation awaiting a recorded CARLA run.
Phases 4 onward have not been started.

## Phase 1: metric camera-relative 3D localization

Status: **Implemented and validated in a controlled CARLA scenario**.
Evidence: [verified report](results/phase1_localization_verified.json). Frames 5226–5235
passed consecutively for track 2 / actor 25. Mean position error 0.04470 m,
maximum 0.07170 m; 183 preceding WAIT frames did not count toward success.

The existing `ObjectLocalizer` still supplies image position and motion. After
`SensorFusion.fuse_rgbd`, `localize_rgbd` back-projects valid central-ROI depth
pixels and takes a coordinate-wise median. Original `distance_m`, track, region,
motion and prediction fields are preserved. No world-coordinate localization or
actor-center estimate is claimed: the result represents the visible surface
inside the detection ROI. Occluders/background may still contaminate a box.

For width W, height H, horizontal FOV theta and square pixels:

```text
fx = fy = W / (2 * tan(theta/2))
cx = W/2; cy = H/2
X = (u-cx)*Z/fx; Y = (v-cy)*Z/fy; Z = metric axial depth
```

Optical coordinates are **X right, Y down, Z forward**, in meters. Integer pixel
indices follow the existing CARLA projection convention. The test cameras use
zero lens distortion. CARLA LiDAR axes map as `(X,Y,Z) = (lidar.y,-lidar.z,lidar.x)`.

Added fields: `position_3d_camera_m`, `x_m`, `y_m`, `z_m`,
`localization_valid`, `localization_source="CARLA_RGBD"`,
`localization_frame_id`, `localization_sample_count`, `localization_reference`.
Invalid/stale/untracked inputs produce null coordinates and valid=false.

## Run Phase 1

Start CARLA without another synchronous tick owner, then run:

```powershell
.\.venv\Scripts\python.exe carla_main.py
```

Select `TEST_MODE = "RGBD_3D_LOCALIZATION_TEST"` to repeat Phase 1. The existing RGB-D frame buffer,
co-located semantic LiDAR, independent NPC association and 20 Hz synchronous
validation runner are reused. Association requires actor-labelled coverage and
rejects ambiguous detections. Reference XYZ is the median NPC surface position
in the same central ROI, from independent LiDAR samples in the same frame.
Different raster/ray sampling and collision/render geometry can cause differences.

Acceptance thresholds are fixed before live evaluation: absolute X and Y error
<= 0.35 m each, Z error <= 0.60 m, Euclidean error <= 0.75 m. All must pass for
10 distinct consecutive frames. WAIT/FAIL resets the streak. Timeout or 200
attempts without the required sequence never counts as success.

Required terminal result: `RGBD_3D_LOCALIZATION_TEST_PASSED`.
`rgbd_3d_localization_report.json` saves track/actor/frame IDs, predicted/reference
XYZ, per-axis and Euclidean errors, thresholds and every frame outcome.
The old distance report is not overwritten.

For this staged change, 3D augmentation runs in the two pinhole RGB-D validation
modes. NORMAL and the existing recovery/dropout/traffic-light modes retain their
camera settings and behavior. Broader activation is deferred until this phase
passes, rather than applying a pinhole model to distorted images.

Local checks (not live validation):

```powershell
.\.venv\Scripts\python.exe -m unittest test_rgbd test_rgbd_validation test_metric_localization -v
```

## Evidence and remaining phases

| Capability | Status |
|---|---|
| RGB-D sensor fusion | Validated in the supplied controlled CARLA run; 10 frames, 0.150–0.273 m observed error |
| Traffic-light validation | Validated in supplied RED/YELLOW/GREEN run |
| Metric 3D localization | Validated in supplied controlled run; mean 4.47 cm, max 7.17 cm |
| Global route planning | Validated: 22/22 points, arrival 2.28 m from destination |
| Quantitative trajectory evaluation | Runner implemented; awaiting CARLA observations and results |
| System benchmark, C++ component, deployment | Future work |

See [RGBD.md](RGBD.md) for the existing distance validation. Controlled tests do
not establish real-world or all-scenario performance. EdgeTrack artifacts have
not been modified as part of Phase 1.


## Phase 2: destination-based navigation (validated controlled run)

Select `TEST_MODE = "ROUTE_NAVIGATION_TEST"` to repeat. Existing `WaypointNavigator` remains intact.
`RouteNavigator` uses the installed CARLA GlobalRoutePlanner at 2 m resolution,
selects a reachable destination, consumes route waypoints in order, and supplies
steering to the existing vehicle controller. Object/traffic-light/safety/agent
logic still controls throttle and brake. No autopilot or manual input is used.
This is waypoint navigation, not obstacle-avoidance path replanning.

Configure `ROUTE_START_INDEX` (default 0) and `ROUTE_DESTINATION_INDEX` (None for
nearest suitable reachable destination, or explicit map spawn index). An occupied
explicit start fails rather than silently changing the scenario. Route acceptance:
40–200 m, at least ten waypoints, ordered waypoint progress, and final destination
within 3 m after reaching the final route segment. Timeout: 180 seconds, including
sensor waits. Success stops the vehicle and prints `ROUTE_NAVIGATION_TEST_PASSED`.
Results are saved to `results/route_navigation_report.json` on exit.

```powershell
$env:CARLA_ROOT = "D:\CARLA_0.9.16"
.\.venv\Scripts\python.exe carla_main.py
```

The local CARLA 0.9.16 planner import was checked. The simulator and sensor
pipeline must run to validate route completion. Traffic or perception STOP may
prevent completion; the test does not bypass them to manufacture a PASS.
Phase 2 passed in the supplied run: 22/22 waypoints, destination distance 2.28 m.
Evidence: [route report](results/phase2_route_verified.json).

The first live route run timed out: all 22 planner points were consumed while
the vehicle was still outside the 3 m destination radius (closest approach
3.148 m). The failed report is retained in
`results/route_navigation_failed_before_endpoint_fix.json`. The route now trims
the planner's trailing segment at the closest destination-lane waypoint within
4 m, then adds the actual destination as the final target. Failure to connect
the route is rejected; exhausted routes outside the arrival radius stop and
fail rather than steering back toward an old waypoint. The 3 m acceptance radius
is unchanged. The corrected implementation passed the subsequent supplied live run.


## Phase 3: quantitative trajectory evaluation

Default mode: `TRAJECTORY_PREDICTION_TEST`. Run `python carla_main.py` with CARLA
open. The existing RGB-D/perception/control path collects 200 processed frames
(up to 180 seconds), using a stationary NPC ahead of the moving ego. Start index
is 0. This is one controlled scene, not a representative motion benchmark.
The existing predictor is unchanged except for exposing its intermediate future
centers: mean image-space velocity, static gating and direction corrections.

Results under `results/`:
- `trajectory_observations.jsonl`: predictions frozen at observation time, track
  centers, class, frame and simulation timestamp (including empty observations).
- `trajectory_metadata.json`: map, scenario and predictor settings.
- `trajectory_prediction_results.csv`: paired constant-position/current-predictor
  ADE and FDE for every eligible scenario/frame/track.
- `trajectory_prediction_summary.json`: mean and P95 ADE/FDE for both models.

Metrics are **pixels**, over five future processed observations. Actual horizon
seconds are recorded because asynchronous processing has variable cadence.
Future tracked centers are observation ground truth, not physical actor truth;
tracking jitter, ID switches and ego motion affect these metrics. Require five
continuous past observations and a complete five-observation future with the same
track/class. Gaps are excluded. Both models use identical eligible windows.
At least 20 eligible windows produce `TRAJECTORY_PREDICTION_EVALUATED`; otherwise
`TRAJECTORY_PREDICTION_INSUFFICIENT_DATA`. Evaluation does not assert improvement.
Overlapping windows are correlated; results are descriptive, not confidence bounds.

Repeat evaluation without CARLA (recorded observations make the evaluation reproducible):

```powershell
python -m evaluation.trajectory results/trajectory_observations.jsonl --output results/replay
```

No measured trajectory numbers have been claimed yet. Do not begin Phase 4 until
the actual user-run evaluation artifacts have been checked.


## C++17 RGB-D Localization Optimization

SmartDrive-Mini includes a native C++17 implementation of RGB-D
visible-surface localization, alongside the Python/NumPy reference
implementation.

### Optimization Techniques

- Replaced full sorting with `std::nth_element` for median selection.
- Preallocated coordinate vectors using `std::vector::reserve()`.
- Reduced repeated camera-intrinsics validation inside pixel loops.
- Precomputed reciprocal focal lengths.
- Reduced repeated calculations during 3D back-projection.

### Performance Benchmark

The benchmark uses a deterministic synthetic depth image
(640 × 480), a single bounding box, 2,000 measured iterations,
and 200 warmup iterations.

| Metric | Python / NumPy | Optimized C++17 |
|---|---:|---:|
| Mean Latency | 0.6905 ms | 0.2711 ms |
| P95 Latency | 0.8513 ms | 0.3112 ms |
| Throughput | 1,448.23 ops/s | 3,689.10 ops/s |
| Relative Speedup | 1.00x | 2.55x |

**Numerical parity:** PASSED

**Position error:** 0.0000000000 m

### Verification

- C++ geometry unit tests: PASSED
- Python/C++ parity tests: PASSED
- CTest: 2/2 tests passed

### Reproduce the Benchmark

Build the project in Release mode:

```cmd
cmake --build cpp\build --config Release
```

Run the tests:

```cmd
ctest --test-dir cpp\build -C Release --output-on-failure
```

Run the performance comparison:

```cmd
python cpp\benchmarks\compare_performance.py cpp\build\Release\geometry_benchmark.exe
```

Results are saved to:

`results/rgbd_cpp_python_performance.json`

### Benchmark Scope

This is a microbenchmark of the RGB-D localization algorithm,
not a measurement of full CARLA simulation performance.

The benchmark excludes Python-to-C++ integration overhead,
camera acquisition, object detection, tracking, and vehicle control.
