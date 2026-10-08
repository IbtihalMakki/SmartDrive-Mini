from ultralytics import YOLO


class ObjectTracker:
    def __init__(self, model_name="yolo11n.pt", device=0):
        self.model = YOLO(model_name)
        self.device = device

    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            device=self.device,
            conf=0.4,
            verbose=False
        )

        return results[0]