import tempfile
import unittest
from evaluation.trajectory import evaluate
from autonomy.prediction import TrajectoryPredictor


def records(count=40):
    return [dict(frame=i+1, timestamp=i*.1, scenario="linear", objects=[
        dict(track_id=1, class_name="car", center=[i, 0],
             predicted_centers=[[i+k, 0] for k in range(1, 6)])
    ]) for i in range(count)]


class TrajectoryEvaluationTests(unittest.TestCase):
    def test_known_ade_fde(self):
        with tempfile.TemporaryDirectory() as output:
            result = evaluate(records(), output)
        self.assertEqual(result["status"], "EVALUATED")
        self.assertEqual(result["models"]["current_predictor"]["mean_ade_px"], 0)
        self.assertEqual(result["models"]["constant_position"]["mean_ade_px"], 3)
        self.assertEqual(result["models"]["constant_position"]["mean_fde_px"], 5)

    def test_missing_track_excludes_windows(self):
        data = records(15)
        data[7]["objects"] = []
        with tempfile.TemporaryDirectory() as output:
            result = evaluate(data, output)
        self.assertEqual(result["eligible_windows"], 0)
        self.assertEqual(result["status"], "INSUFFICIENT_DATA")

    def test_duplicate_frames_rejected(self):
        data = records()
        data[1]["frame"] = data[0]["frame"]
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaises(ValueError):
                evaluate(data, output)

    def test_predictor_path_preserves_original_endpoint(self):
        predictor = TrajectoryPredictor()
        for i in range(6):
            result = predictor.predict([dict(track_id=1, center=[i*6, 2], motion="RIGHT")])[0]
            self.assertEqual(result["predicted_centers"][-1], result["predicted_center"])


if __name__ == "__main__":
    unittest.main()
