import numpy as np


class SensorFusion:
    def __init__(self):
        pass

    def simulate_distance(self, obj, frame_height):
        """
        Simulate a range sensor measurement.

        This is NOT a real physical distance measurement.
        We use the bottom of the bounding box as a simple
        proxy: objects lower in the image are treated as closer.
        """

        x1, y1, x2, y2 = obj["bbox"]

        # Bottom position of the object
        bottom_y = y2

        # Normalize between 0 and 1
        normalized_bottom = bottom_y / frame_height

        # Simulated range:
        # high object in image -> farther away
        # low object in image  -> closer
        min_distance = 2.0
        max_distance = 30.0

        distance_m = max_distance - (
            normalized_bottom *
            (max_distance - min_distance)
        )

        # Keep value inside simulation limits
        distance_m = max(
            min_distance,
            min(max_distance, distance_m)
        )

        return round(distance_m, 2)

    def fuse(self, objects, frame_height):
        fused_objects = []

        for obj in objects:

            simulated_distance = self.simulate_distance(
                obj,
                frame_height
            )

            fused = obj.copy()

            fused.update({
                "distance_m": simulated_distance,
                "distance_source": "SIMULATED_RANGE_SENSOR",
                "camera_detected": True,
                "range_sensor_detected": True
            })

            fused_objects.append(fused)

        return fused_objects

    def fuse_rgbd(self, objects, depth_m, frame_id):
        """Estimate camera-relative surface depth from the central 50% of a box.

        Median sampling reduces edge/background contamination, but does not
        prove object association under occlusion. No synthetic fallback is used.
        """
        if depth_m.ndim != 2:
            raise ValueError("Expected a metric H x W depth map")
        height, width = depth_m.shape
        fused_objects = []
        for obj in objects:
            box = np.asarray(obj["bbox"], dtype=float)
            samples = np.array([], dtype=np.float32)
            valid_fraction = 0.0
            if box.shape == (4,) and np.isfinite(box).all():
                x1, y1, x2, y2 = box
                if x2 > x1 and y2 > y1:
                    dx, dy = (x2 - x1) * 0.25, (y2 - y1) * 0.25
                    left = int(np.clip(np.floor(x1 + dx), 0, width))
                    right = int(np.clip(np.ceil(x2 - dx), 0, width))
                    top = int(np.clip(np.floor(y1 + dy), 0, height))
                    bottom = int(np.clip(np.ceil(y2 - dy), 0, height))
                    roi = depth_m[top:bottom, left:right]
                    samples = roi[np.isfinite(roi) & (roi > 0) & (roi < 999.0)]
                    valid_fraction = samples.size / roi.size if roi.size else 0.0
            valid = samples.size >= 4 and valid_fraction >= 0.5
            fused = obj.copy()
            fused.update({
                "distance_m": round(float(np.median(samples)), 2) if valid else float("nan"),
                "distance_source": "CARLA_DEPTH" if valid else "CARLA_DEPTH_INVALID",
                "distance_valid": valid,
                "distance_reference": "CAMERA_SURFACE_DEPTH",
                "depth_frame_id": frame_id,
                "depth_valid_fraction": round(valid_fraction, 3),
                "depth_sample_count": int(samples.size),
                "camera_detected": True,
                "range_sensor_detected": valid,
            })
            fused_objects.append(fused)
        return fused_objects
