"""Independent semantic ray-cast surface reference, used only for validation."""
import numpy as np


SEMANTIC_POINT = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("cos", "<f4"), ("actor", "<u4"), ("tag", "<u4"),
])


def validate_distance(objects, measurement, npc_id, width, height, fov=90):
    """Compare ROI depth median to independent NPC surface samples.

    Uses camera-forward (axial) depth, not Euclidean slant range. Require
    identity purity and spatial coverage before accepting an association.
    """
    result = {"status": "WAIT", "association": "UNVERIFIED", "reason": "NO_UNIQUE_NPC_DETECTION"}
    points = np.frombuffer(measurement.raw_data, dtype=SEMANTIC_POINT)
    points = points[np.isfinite(points["x"]) & np.isfinite(points["y"])
                    & np.isfinite(points["z"]) & (points["x"] > 0.1)]
    focal = width / (2 * np.tan(np.radians(fov) / 2))
    u = width / 2 + focal * points["y"] / points["x"]
    v = height / 2 - focal * points["z"] / points["x"]
    candidates = []
    for obj in objects:
        if obj.get("class_name", "").lower() not in {"car", "truck", "bus", "motorcycle"}:
            continue
        box = np.asarray(obj["bbox"], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        dx, dy = (x2-x1)*0.25, (y2-y1)*0.25
        left, right = np.clip([x1+dx, x2-dx], 0, width)
        top, bottom = np.clip([y1+dy, y2-dy], 0, height)
        roi = (u >= left) & (u < right) & (v >= top) & (v < bottom)
        npc = roi & (points["actor"] == npc_id)
        count = int(npc.sum())
        if count < 20 or count / max(int(roi.sum()), 1) < 0.9:
            continue
        if np.ptp(u[npc]) < (right-left)*0.5 or np.ptp(v[npc]) < (bottom-top)*0.5:
            continue
        candidates.append((obj, float(np.median(points["x"][npc])), count))
    if len(candidates) != 1:
        return result
    obj, reference, count = candidates[0]
    result.update(association="PASS", track_id=obj.get("track_id"), npc_id=npc_id,
                  reference_m=reference, samples=count, frame=measurement.frame)
    measured = obj.get("distance_m", float("nan"))
    if (not obj.get("distance_valid") or not np.isfinite(measured)
            or obj.get("depth_frame_id") != measurement.frame):
        result.update(status="FAIL", reason="INVALID_OR_MISMATCHED_DEPTH")
        return result
    error = abs(measured-reference)
    tolerance = max(0.5, 0.05*reference)
    result.update(status="PASS" if error <= tolerance else "FAIL",
                  reason="SURFACE_COMPARISON", depth_m=measured,
                  error_m=error, tolerance_m=tolerance)
    return result


class DistanceValidationRun:
    """Require ten distinct consecutive passing frames; keep every result."""
    def __init__(self):
        self.results = []
        self.streak = 0
        self.passed = False

    def record(self, frame, result):
        if self.results and frame <= self.results[-1]["frame"]:
            return
        self.results.append(dict(result, frame=frame))
        self.streak = self.streak + 1 if result["status"] == "PASS" else 0
        self.passed = self.streak >= 10
