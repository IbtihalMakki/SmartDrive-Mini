from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_name="yolo11n.pt", device=0):
        self.model = YOLO(model_name)
        self.device = device

    def detect(self, frame):
        results = self.model(
            frame,
            device=self.device,
            verbose=False
        )

        return results[0]