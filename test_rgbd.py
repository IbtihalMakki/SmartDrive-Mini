"""RGB-D data-path checks; no CARLA server or neural models required."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from simulation.rgbd import RGBDFrameBuffer, decode_depth
from autonomy.sensor_fusion import SensorFusion
from autonomy.risk_assessment import RiskAssessor


def image(frame=1, meters=12.5, width=8, height=8):
    code = round(meters / 1000 * 16777215)
    pixel = [code >> 16 & 255, code >> 8 & 255, code & 255, 255]
    raw = np.tile(np.array(pixel, dtype=np.uint8), (height, width, 1))
    return SimpleNamespace(frame=frame, timestamp=frame * 0.05,
                           width=width, height=height, raw_data=raw.tobytes())


class RGBDTests(unittest.TestCase):
    def test_metric_decoding_channel_order(self):
        for meters in (0, 1, 12.5, 100, 1000):
            np.testing.assert_allclose(decode_depth(image(meters=meters)),
                                       meters, atol=0.0002)

    def test_only_matching_frames_consumed_once(self):
        buffer = RGBDFrameBuffer()
        buffer.add("rgb", image(2))
        buffer.add("depth", image(1))
        self.assertIsNone(buffer.pop())
        buffer.add("depth", image(2))
        frame, timestamp, rgb, depth = buffer.pop()
        self.assertEqual(frame, 2)
        self.assertEqual(rgb.shape, (8, 8, 3))
        np.testing.assert_allclose(depth, 12.5, atol=0.0002)
        self.assertIsNone(buffer.pop())
        buffer.add("rgb", image(2))
        buffer.add("depth", image(2))
        self.assertIsNone(buffer.pop())

    def test_buffer_bounded_and_stale_frames_rejected(self):
        buffer = RGBDFrameBuffer(capacity=2, max_age=2)
        with patch("simulation.rgbd.time.monotonic", return_value=10):
            for frame in range(5):
                buffer.add("rgb", image(frame))
            buffer.add("depth", image(4))
        self.assertEqual(len(buffer.frames["rgb"]), 2)
        with patch("simulation.rgbd.time.monotonic", return_value=13):
            self.assertIsNone(buffer.pop())

    def test_mismatched_geometry_rejected(self):
        buffer = RGBDFrameBuffer()
        buffer.add("rgb", image(width=8))
        buffer.add("depth", image(width=4))
        with self.assertRaises(ValueError):
            buffer.pop()

    def test_box_center_avoids_background_and_keeps_track(self):
        depth = np.full((8, 8), 100, dtype=np.float32)
        depth[2:6, 2:6] = 7.5
        depth[2, 2] = 999.9
        obj = {"bbox": [0, 0, 8, 8], "track_id": 42}
        fused = SensorFusion().fuse_rgbd([obj], depth, 9)[0]
        self.assertEqual(fused["distance_m"], 7.5)
        self.assertEqual(fused["track_id"], 42)
        self.assertEqual(fused["depth_frame_id"], 9)
        self.assertEqual(fused["distance_source"], "CARLA_DEPTH")

    def test_invalid_depth_has_no_synthetic_fallback(self):
        for box in ([0, 0, 8, 8], [-20, -20, -10, -10], [5, 5, 1, 1]):
            obj = {"bbox": box, "region": "CENTER", "motion": "STATIC"}
            fused = SensorFusion().fuse_rgbd([obj], np.zeros((8, 8)), 1)
            self.assertFalse(fused[0]["distance_valid"])
            self.assertTrue(np.isnan(fused[0]["distance_m"]))
            self.assertEqual(RiskAssessor().assess(fused)[0]["recommended_action"], "STOP")


if __name__ == "__main__":
    unittest.main()
