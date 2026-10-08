# SmartDrive-Mini — C++17 RGB-D Geometry (Phase 5)

Standalone, dependency-free C++17 implementation of the geometry in `autonomy/metric_localization.py`.

- CARLA horizontal FOV pinhole intrinsics: `fx=fy=width/(2*tan(hfov/2))`.
- Camera optical coordinates: X right, Y down, Z forward, meters.
- Central-half bounding box ROI; per-axis median of valid pixel back-projections.
- Valid depth `0 < z < 999`, finite; at least 4 valid samples and 50% ROI coverage.
- Returns visible-surface representative point **not** actor/world centroid.
- Geometry-only API: tracking/frame synchronization/`distance_valid` guards remain in Python.

## Build & test (Windows PowerShell, from repository root)

```powershell
cmake -S cpp -B cpp/build
cmake --build cpp/build --config Release
ctest --test-dir cpp/build -C Release --output-on-failure
```

If `cmake` or a C++ compiler is unavailable, install Visual Studio Build Tools with Desktop development with C++ and CMake.

## Validation scope

C++ unit tests and Python/C++ numerical parity on **synthetic fixtures**. Not integrated into the live CARLA runtime; Phase 4 evidence remains unchanged.

Add `cpp/build/` to the repository `.gitignore` before committing.
