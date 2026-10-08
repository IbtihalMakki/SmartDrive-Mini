from perception.tracking import ObjectTracker
from perception.segmentation import Segmenter
from perception.pose import PoseEstimator


class PerceptionPipeline:
    def __init__(self):
        self.tracker = ObjectTracker()
        self.segmenter = Segmenter()
        self.pose_estimator = PoseEstimator()

    def process(self, frame):
        tracking_result = self.tracker.track(frame)
        segmentation_result = self.segmenter.segment(frame)
        pose_result = self.pose_estimator.estimate(frame)

        return {
            "tracking": tracking_result,
            "segmentation": segmentation_result,
            "pose": pose_result
        }