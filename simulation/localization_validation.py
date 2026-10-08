"""3D surface-position validation against independent actor-labelled LiDAR."""
import numpy as np
from autonomy.metric_localization import camera_intrinsics, central_roi
from simulation.rgbd_validation import SEMANTIC_POINT, validate_distance

# Fixed before any live evaluation; do not adapt to observed errors.
AXIS_TOLERANCE_M = [0.35, 0.35, 0.60]
POSITION_TOLERANCE_M = 0.75


def validate_localization(objects, measurement, npc_id, width, height, fov=90):
    # Reuse the established independent actor association, including ambiguity,
    # purity, count and ROI coverage gates. Never select by nearest distance.
    association = validate_distance(objects, measurement, npc_id, width, height, fov)
    result = dict(status="WAIT", association=association["association"],
                  reason=association["reason"], frame=measurement.frame, npc_id=npc_id,
                  axis_tolerance_m=AXIS_TOLERANCE_M[:], position_tolerance_m=POSITION_TOLERANCE_M)
    if association["association"] != "PASS":
        return result
    track = association["track_id"]
    matches = [obj for obj in objects if obj.get("track_id") == track]
    result["track_id"] = track
    if len(matches) != 1 or track is None:
        result.update(status="WAIT", reason="AMBIGUOUS_TRACK")
        return result
    obj = matches[0]
    predicted = np.asarray(obj.get("position_3d_camera_m"), dtype=float)
    if (not obj.get("localization_valid") or predicted.shape != (3,)
            or not np.isfinite(predicted).all()
            or obj.get("localization_frame_id") != measurement.frame
            or obj.get("depth_frame_id") != measurement.frame):
        result.update(status="FAIL", reason="INVALID_OR_STALE_LOCALIZATION")
        return result
    points = np.frombuffer(measurement.raw_data, dtype=SEMANTIC_POINT)
    valid = ((points["actor"] == npc_id) & (points["x"] > 0.1)
             & np.isfinite(points["x"]) & np.isfinite(points["y"]) & np.isfinite(points["z"]))
    points = points[valid]
    k = camera_intrinsics(width, height, fov)
    u = k["cx"] + k["fx"]*points["y"]/points["x"]
    v = k["cy"] - k["fy"]*points["z"]/points["x"]
    left, top, right, bottom = central_roi(obj["bbox"], width, height)
    points = points[(u >= left) & (u < right) & (v >= top) & (v < bottom)]
    if len(points) < 20:
        result.update(reason="INSUFFICIENT_REFERENCE_POINTS")
        return result
    # CARLA sensor axes (forward,right,up) -> optical (right,down,forward).
    reference = np.array([np.median(points["y"]), -np.median(points["z"]), np.median(points["x"])])
    error = np.abs(predicted-reference)
    norm = float(np.linalg.norm(error))
    passed = bool(np.all(error <= AXIS_TOLERANCE_M) and norm <= POSITION_TOLERANCE_M)
    result.update(status="PASS" if passed else "FAIL", reason="3D_SURFACE_COMPARISON",
                  predicted_xyz_m=predicted.tolist(), reference_xyz_m=reference.tolist(),
                  axis_absolute_error_m=error.tolist(), position_error_m=norm,
                  reference_samples=len(points))
    return result
