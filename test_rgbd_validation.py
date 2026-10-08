import unittest
from types import SimpleNamespace
import numpy as np
from simulation.rgbd_validation import SEMANTIC_POINT, validate_distance, DistanceValidationRun


class DistanceValidationTests(unittest.TestCase):
    def setUp(self):
        self.points = np.zeros(100, dtype=SEMANTIC_POINT)
        self.points["x"] = 10
        yy, zz = np.meshgrid(np.linspace(-1.8, 1.8, 10), np.linspace(-1.8, 1.8, 10))
        self.points["y"] = yy.ravel()
        self.points["z"] = zz.ravel()
        self.points["actor"] = 42
        self.obj = dict(bbox=[30, 30, 70, 70], class_name="car", track_id=5,
                        distance_valid=True, distance_m=10.1, depth_frame_id=7)

    def validate(self, objects=None):
        return validate_distance([self.obj] if objects is None else objects,
                                 SimpleNamespace(raw_data=self.points.tobytes(), frame=7),
                                 42, 100, 100)

    def test_independent_surface_and_actor_match(self):
        result = self.validate()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["reference_m"], 10)
        self.assertAlmostEqual(result["error_m"], 0.1)

    def test_wrong_actor_and_occlusion_cannot_pass(self):
        self.points["actor"][:30] = 99
        self.assertEqual(self.validate()["status"], "WAIT")
        self.points["actor"] = 99
        self.assertEqual(self.validate()["association"], "UNVERIFIED")

    def test_depth_error_and_frame_mismatch_fail(self):
        self.obj["distance_m"] = 13.31
        self.assertEqual(self.validate()["status"], "FAIL")
        self.obj["distance_m"] = 10
        self.obj["depth_frame_id"] = 6
        self.assertEqual(self.validate()["status"], "FAIL")

    def test_ambiguous_detections_do_not_pass(self):
        self.assertEqual(self.validate([self.obj, dict(self.obj, track_id=6)])["status"], "WAIT")

    def test_insufficient_coverage_does_not_pass(self):
        self.points["y"] = 0
        self.assertEqual(self.validate()["status"], "WAIT")

    def test_run_requires_distinct_consecutive_frames(self):
        run = DistanceValidationRun()
        for frame in range(9):
            run.record(frame, {"status": "PASS"})
        run.record(8, {"status": "PASS"})
        self.assertFalse(run.passed)
        run.record(9, {"status": "WAIT"})
        self.assertEqual(run.streak, 0)
        for frame in range(10, 20):
            run.record(frame, {"status": "PASS"})
        self.assertTrue(run.passed)


if __name__ == "__main__":
    unittest.main()
