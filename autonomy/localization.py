class ObjectLocalizer:
    def __init__(self):
        self.previous_positions = {}

    def localize(self, objects, frame_width, frame_height):
        localized_objects = []

        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]

            # Center of bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Normalize position to 0-1
            norm_x = center_x / frame_width
            norm_y = center_y / frame_height

            # Horizontal region
            if norm_x < 0.33:
                region = "LEFT"
            elif norm_x > 0.66:
                region = "RIGHT"
            else:
                region = "CENTER"

            # Estimate motion using previous position
            track_id = obj["track_id"]
            motion = "UNKNOWN"
            dx = 0
            dy = 0

            if track_id is not None and track_id in self.previous_positions:
                prev_x, prev_y = self.previous_positions[track_id]

                dx = center_x - prev_x
                dy = center_y - prev_y

                threshold = 5

                if abs(dx) < threshold and abs(dy) < threshold:
                    motion = "STATIC"
                elif abs(dx) > abs(dy):
                    motion = "RIGHT" if dx > 0 else "LEFT"
                else:
                    motion = "DOWN" if dy > 0 else "UP"

            if track_id is not None:
                self.previous_positions[track_id] = (
                    center_x,
                    center_y
                )

            localized = obj.copy()

            localized.update({
                "center": [
                    round(center_x, 1),
                    round(center_y, 1)
                ],
                "normalized_position": [
                    round(norm_x, 3),
                    round(norm_y, 3)
                ],
                "region": region,
                "motion": motion,
                "dx": round(dx, 1),
                "dy": round(dy, 1)
            })

            localized_objects.append(localized)

        return localized_objects