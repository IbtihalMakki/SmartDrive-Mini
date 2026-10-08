from ultralytics import YOLO


class PoseEstimator:
    def __init__(self, model_name="yolo11n-pose.pt", device=0):
        self.model = YOLO(model_name)
        self.device = device

    def estimate(self, frame):
        results = self.model(
            frame,
            device=self.device,
            conf=0.4,
            verbose=False
        )

        return results[0]