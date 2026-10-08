import carla


class TrafficLightHandler:
    """
    Handles traffic-light semantics using CARLA ground truth.

    Traffic lights are NOT physical obstacles for the
    generic obstacle RiskAssessor. Their STATE determines
    the driving decision.
    """

    def __init__(
        self,
        vehicle,
        world,
        max_detection_distance=35.0
    ):
        self.vehicle = vehicle
        self.world = world

        self.max_detection_distance = (
            max_detection_distance
        )

    # =========================================================
    # STATE NAME
    # =========================================================

    @staticmethod
    def _state_to_string(state):

        if state == carla.TrafficLightState.Red:
            return "RED"

        if state == carla.TrafficLightState.Yellow:
            return "YELLOW"

        if state == carla.TrafficLightState.Green:
            return "GREEN"

        if state == carla.TrafficLightState.Off:
            return "OFF"

        return "UNKNOWN"

    # =========================================================
    # CURRENT AFFECTING TRAFFIC LIGHT
    # =========================================================

    def get_status(self):
        """
        Return traffic-light ground truth for the ego vehicle.

        CARLA's vehicle traffic-light API is preferred because
        it tells us which traffic light is actually affecting
        the ego vehicle, rather than simply choosing any traffic
        light visible in the camera.
        """

        try:
            traffic_light = (
                self.vehicle.get_traffic_light()
            )

        except Exception:
            traffic_light = None

        if traffic_light is None:

            return {
                "detected": False,
                "state": "NONE",
                "distance_m": None,
                "actor_id": None,
                "risk": "LOW",
                "action": "CONTINUE"
            }

        ego_location = (
            self.vehicle.get_location()
        )

        light_location = (
            traffic_light.get_location()
        )

        distance = ego_location.distance(
            light_location
        )

        try:
            state = (
                traffic_light.get_state()
            )

        except Exception:
            state = (
                carla.TrafficLightState.Unknown
            )

        state_name = (
            self._state_to_string(state)
        )

        # -----------------------------------------------------
        # Too far away
        # -----------------------------------------------------

        if (
            distance
            > self.max_detection_distance
        ):

            return {
                "detected": False,
                "state": state_name,
                "distance_m": distance,
                "actor_id": traffic_light.id,
                "risk": "LOW",
                "action": "CONTINUE"
            }

        # -----------------------------------------------------
        # Semantic traffic-light decision
        # -----------------------------------------------------

        if state_name == "RED":

            risk = "HIGH"
            action = "STOP"

        elif state_name == "YELLOW":

            risk = "MEDIUM"
            action = "SLOW_DOWN"

        elif state_name == "GREEN":

            risk = "LOW"
            action = "CONTINUE"

        else:

            risk = "LOW"
            action = "CONTINUE"

        return {
            "detected": True,
            "state": state_name,
            "distance_m": distance,
            "actor_id": traffic_light.id,
            "risk": risk,
            "action": action
        }