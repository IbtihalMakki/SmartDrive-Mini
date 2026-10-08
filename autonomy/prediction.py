from collections import defaultdict, deque


class TrajectoryPredictor:
    def __init__(self, history_size=5, prediction_steps=5):
        self.history = defaultdict(
            lambda: deque(maxlen=history_size)
        )

        self.prediction_steps = prediction_steps

    def predict(self, objects):
        predicted_objects = []

        for obj in objects:
            track_id = obj["track_id"]

            if track_id is None:
                continue

            current_x, current_y = obj["center"]
            motion = obj["motion"]

            # Store current position
            self.history[track_id].append(
                (current_x, current_y)
            )

            history = self.history[track_id]

            # Default:
            # no movement prediction
            predicted_x = current_x
            predicted_y = current_y

            avg_dx = 0.0
            avg_dy = 0.0

            # Only predict movement if the object
            # is currently classified as moving
            if len(history) >= 2 and motion != "STATIC":

                dx_values = []
                dy_values = []

                for i in range(1, len(history)):
                    dx = history[i][0] - history[i - 1][0]
                    dy = history[i][1] - history[i - 1][1]

                    dx_values.append(dx)
                    dy_values.append(dy)

                avg_dx = sum(dx_values) / len(dx_values)
                avg_dy = sum(dy_values) / len(dy_values)

                # -----------------------------------------
                # DIRECTION CONSISTENCY
                # -----------------------------------------

                # If current motion says RIGHT,
                # do not allow old history to predict LEFT
                if motion == "RIGHT" and avg_dx < 0:
                    avg_dx = max(
                        current_x - history[-2][0],
                        0
                    )

                # If current motion says LEFT,
                # do not allow old history to predict RIGHT
                elif motion == "LEFT" and avg_dx > 0:
                    avg_dx = min(
                        current_x - history[-2][0],
                        0
                    )

                # Same idea for vertical movement
                if motion == "DOWN" and avg_dy < 0:
                    avg_dy = max(
                        current_y - history[-2][1],
                        0
                    )

                elif motion == "UP" and avg_dy > 0:
                    avg_dy = min(
                        current_y - history[-2][1],
                        0
                    )

                # -----------------------------------------
                # FUTURE POSITION
                # -----------------------------------------
                predicted_x = (
                    current_x +
                    avg_dx * self.prediction_steps
                )

                predicted_y = (
                    current_y +
                    avg_dy * self.prediction_steps
                )

            predicted = obj.copy()

            predicted.update({
                "predicted_center": [
                    round(predicted_x, 1),
                    round(predicted_y, 1)
                ],
                "predicted_centers": [
                    [round(current_x + avg_dx * step, 1),
                     round(current_y + avg_dy * step, 1)]
                    for step in range(1, self.prediction_steps + 1)
                ],
                "velocity_px_per_frame": [
                    round(avg_dx, 2),
                    round(avg_dy, 2)
                ]
            })

            predicted_objects.append(predicted)

        return predicted_objects
