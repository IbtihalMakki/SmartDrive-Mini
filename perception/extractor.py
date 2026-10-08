class PerceptionExtractor:

    @staticmethod
    def extract_tracks(result):

        objects = []

        if result.boxes is None:
            return objects

        for box in result.boxes:

            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            track_id = None

            if box.id is not None:
                track_id = int(box.id.item())

            objects.append({
                "track_id": track_id,
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": round(confidence, 3),
                "bbox": [
                    round(x1, 1),
                    round(y1, 1),
                    round(x2, 1),
                    round(y2, 1)
                ]
            })

        return objects