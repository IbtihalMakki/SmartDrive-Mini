"""Camera-relative visible-surface localization; never an actor/world centroid."""
import numpy as np


def camera_intrinsics(width, height, horizontal_fov):
    if width <= 0 or height <= 0 or not 0 < horizontal_fov < 180:
        raise ValueError("Invalid camera dimensions or horizontal FOV")
    focal = width / (2.0 * np.tan(np.radians(horizontal_fov) / 2.0))
    # CARLA pinhole camera has square pixels: fy=fx, not height-based FOV.
    return dict(fx=float(focal), fy=float(focal), cx=width/2.0, cy=height/2.0)


def central_roi(box, width, height):
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        return None
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    dx, dy = (x2-x1)*0.25, (y2-y1)*0.25
    left, right = np.clip([np.floor(x1+dx), np.ceil(x2-dx)], 0, width).astype(int)
    top, bottom = np.clip([np.floor(y1+dy), np.ceil(y2-dy)], 0, height).astype(int)
    return (left, top, right, bottom) if right > left and bottom > top else None


def localize_rgbd(objects, depth_m, frame_id, intrinsics):
    """Median of back-projected valid pixels in the central half-box.

    Optical axes in meters: X right, Y down, Z forward (axial depth).
    This is a representative visible surface position, not object center.
    Existing distance_m and image-space localization are passed through intact.
    """
    if depth_m.ndim != 2:
        raise ValueError("Expected H x W metric depth")
    fx, fy, cx, cy = (intrinsics[k] for k in ("fx", "fy", "cx", "cy"))
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise ValueError("Invalid camera intrinsics")
    height, width = depth_m.shape
    output = []
    for obj in objects:
        result = dict(obj, position_3d_camera_m=None, x_m=None, y_m=None, z_m=None,
                      localization_valid=False, localization_source="CARLA_RGBD",
                      localization_frame_id=frame_id, localization_sample_count=0,
                      localization_reference="VISIBLE_SURFACE_ROI_MEDIAN")
        roi = central_roi(obj["bbox"], width, height)
        if (roi is not None and obj.get("track_id") is not None
                and obj.get("distance_valid") and obj.get("depth_frame_id") == frame_id):
            left, top, right, bottom = roi
            depth = depth_m[top:bottom, left:right]
            valid = np.isfinite(depth) & (depth > 0) & (depth < 999.0)
            count = int(valid.sum())
            if count >= 4 and count / depth.size >= 0.5:
                rows, cols = np.nonzero(valid)
                z = depth[valid].astype(float)
                xyz = [float(np.median((cols+left-cx)*z/fx)),
                       float(np.median((rows+top-cy)*z/fy)), float(np.median(z))]
                result.update(position_3d_camera_m=xyz, x_m=xyz[0], y_m=xyz[1], z_m=xyz[2],
                              localization_valid=True, localization_sample_count=count)
        output.append(result)
    return output
