# SmartDrive-Mini Benchmark

Machine-readable benchmark coverage: **9/9 tests**.

| Test | Status | Evidence / metric |
|---|---|---|
| Metric 3D Localization | PASS | mean error 0.045 m; final 0.026 m |
| Global Route Navigation | PASS | 480.7 m; 237/237 waypoints; final distance 0.866 m |
| Trajectory Prediction | EVALUATED | 353 windows; ADE improvement 46.7%; FDE improvement 40.9% |
| RGB-D Distance | PASS | 10 consecutive PASS frames; reference SEMANTIC_LIDAR_CAMERA_SURFACE |
| Traffic Light — RED | PASS | 30.83 m; HIGH / STOP; brake 1.00 |
| Traffic Light — YELLOW | PASS | 29.48 m; MEDIUM / SLOW_DOWN |
| Traffic Light — GREEN | PASS | 29.14 m; LOW / CONTINUE; throttle 0.35 |
| Perception Dropout Safety | PASS | 1.5 s dropout; latch + override + STOP maintained; safe release verified |
| Autonomous Recovery | PASS | 3.141 s sustained driving; 17.17 km/h at PASS |

## Evidence policy

A test is counted as complete here only when its machine-readable result exists in `results/`. Previously observed console PASS messages are not silently converted into benchmark evidence.

## Current quantitative highlights

- Global route: **480.7 m**, **237/237** waypoints, final destination distance **0.866 m**.
- Trajectory prediction: **353** eligible windows; mean ADE improvement **46.7%** and mean FDE improvement **40.9%** over the constant-position baseline.
- Metric 3D localization: mean PASS-sample position error **0.045 m**, final error **0.026 m**.
- RGB-D distance: **10** consecutive passing frames against **SEMANTIC_LIDAR_CAMERA_SURFACE** reference.
- Perception dropout: **1.5 s** controlled dropout; hazard latch, safety override, STOP persistence, restoration, and safe release all verified.
- Autonomous recovery: **3.141 s** sustained driving after obstacle clearance; **17.17 km/h** at PASS.
- Traffic-light validation uses CARLA semantic traffic-light state (RED/YELLOW/GREEN), not visual color classification.
