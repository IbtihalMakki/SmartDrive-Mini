import unittest
from types import SimpleNamespace
import numpy as np
from autonomy.metric_localization import camera_intrinsics, localize_rgbd
from autonomy.sensor_fusion import SensorFusion
from simulation.localization_validation import validate_localization
from simulation.rgbd_validation import SEMANTIC_POINT


class MetricLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.depth = np.full((100, 100), 10.0)
        self.obj = dict(bbox=[50, 40, 90, 80], track_id=5, class_name="car")
        self.fused = SensorFusion().fuse_rgbd([self.obj], self.depth, 7)
        self.k = camera_intrinsics(100, 100, 90)

    def localized(self):
        return localize_rgbd(self.fused, self.depth, 7, self.k)

    def reference(self, actor=42):
        points = np.zeros(100, dtype=SEMANTIC_POINT)
        x, y = np.meshgrid(np.linspace(2.4, 5.4, 10), np.linspace(.4, 3.4, 10))
        points["x"] = 10
        points["y"] = x.ravel()
        points["z"] = -y.ravel()
        points["actor"] = actor
        return SimpleNamespace(raw_data=points.tobytes(), frame=7)

    def test_non_square_intrinsics(self):
        k = camera_intrinsics(1280, 720, 90)
        np.testing.assert_allclose([k["fx"], k["fy"], k["cx"], k["cy"]], [640, 640, 640, 360])

    def test_off_axis_backprojection_preserves_distance(self):
        result = self.localized()[0]
        np.testing.assert_allclose(result["position_3d_camera_m"], [3.9, 1.9, 10])
        self.assertEqual(result["distance_m"], self.fused[0]["distance_m"])
        self.assertNotIn("position_3d_camera_m", self.fused[0])

    def test_invalid_depth_and_stale_frame(self):
        self.depth[:] = np.nan
        self.assertFalse(self.localized()[0]["localization_valid"])
        self.depth[:] = 10
        self.fused[0]["depth_frame_id"] = 6
        self.assertIsNone(self.localized()[0]["position_3d_camera_m"])

    def test_invalid_boxes_and_missing_track(self):
        for box in ([2, 2, 1, 1], [-30, -30, -10, -10], [0, 0, float("nan"), 9]):
            self.fused[0]["bbox"] = box
            self.assertFalse(self.localized()[0]["localization_valid"])

    def test_independent_3d_reference_and_axis_conversion(self):
        result = validate_localization(self.localized(), self.reference(), 42, 100, 100)
        self.assertEqual(result["status"], "PASS")
        self.assertLess(result["position_error_m"], 1e-5)

    def test_wrong_actor_stale_and_large_axis_error_do_not_pass(self):
        objects = self.localized()
        self.assertEqual(validate_localization(objects, self.reference(99), 42, 100, 100)["status"], "WAIT")
        objects[0]["position_3d_camera_m"][0] += 1
        self.assertEqual(validate_localization(objects, self.reference(), 42, 100, 100)["status"], "FAIL")
        objects[0]["localization_frame_id"] = 6
        self.assertEqual(validate_localization(objects, self.reference(), 42, 100, 100)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
