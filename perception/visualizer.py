import cv2


class PerceptionVisualizer:

    @staticmethod
    def draw(frame, results):

        # 1. Segmentation masks + segmentation boxes
        output = results["segmentation"].plot(
            img=frame.copy(),
            boxes=False,
            labels=False
        )

        # 2. Tracking boxes + IDs
        tracking = results["tracking"]

        if tracking.boxes is not None:
            for box in tracking.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                class_name = tracking.names[class_id]

                track_id = None

                if box.id is not None:
                    track_id = int(box.id.item())

                label = (
                    f"ID {track_id} | "
                    f"{class_name} | "
                    f"{confidence:.2f}"
                )

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    output,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        # 3. Pose keypoints / skeleton
        pose = results["pose"]

        if pose.keypoints is not None:
            pose_overlay = pose.plot(
                img=output,
                boxes=False,
                labels=False
            )

            output = pose_overlay

        return output