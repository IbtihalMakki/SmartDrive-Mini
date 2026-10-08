import time


class DrivingAgent:
    """
    Stateful autonomous driving decision agent.

    Responsibilities:
    - Receive the fused multimodal decision.
    - Maintain driving state across frames.
    - Select the next action.
    - Handle STOP -> WAIT -> REASSESS -> RESUME behavior.
    """

    VALID_STATES = {
        "DRIVING",
        "CAUTION",
        "SLOWING",
        "STOPPED",
        "REASSESSING"
    }

    def __init__(
        self,
        stop_hold_seconds=3.0,
        safe_frames_required=3
    ):
        self.state = "DRIVING"

        self.stop_hold_seconds = stop_hold_seconds
        self.safe_frames_required = safe_frames_required

        self.stop_started_at = None
        self.safe_frame_count = 0

        self.last_action = "CONTINUE"

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------
    def step(self, final_decision):

        risk = str(
            final_decision.get(
                "final_risk",
                "UNKNOWN"
            )
        ).upper()

        action = str(
            final_decision.get(
                "final_action",
                "UNKNOWN"
            )
        ).upper()

        previous_state = self.state

        # -------------------------------------------------
        # HIGH RISK / STOP
        # -------------------------------------------------
        if risk == "HIGH" or action == "STOP":

            self._enter_stop()

            agent_action = "STOP"

            reason = (
                "Safety-critical risk detected. "
                "Agent enters STOPPED state."
            )

        # -------------------------------------------------
        # ALREADY STOPPED
        # -------------------------------------------------
        elif self.state == "STOPPED":

            (
                agent_action,
                reason
            ) = self._handle_stopped_state(
                risk,
                action
            )

        # -------------------------------------------------
        # REASSESSING
        # -------------------------------------------------
        elif self.state == "REASSESSING":

            (
                agent_action,
                reason
            ) = self._handle_reassessment(
                risk,
                action
            )

        # -------------------------------------------------
        # MEDIUM RISK
        # -------------------------------------------------
        elif (
            risk == "MEDIUM"
            or action == "SLOW_DOWN"
        ):

            self.state = "SLOWING"
            self.safe_frame_count = 0

            agent_action = "SLOW_DOWN"

            reason = (
                "Moderate risk detected. "
                "Agent reduces speed and continues monitoring."
            )

        # -------------------------------------------------
        # CAUTION
        # -------------------------------------------------
        elif action == "CAUTION":

            self.state = "CAUTION"
            self.safe_frame_count = 0

            agent_action = "CAUTION"

            reason = (
                "Scene requires caution. "
                "Agent continues with increased monitoring."
            )

        # -------------------------------------------------
        # LOW RISK
        # -------------------------------------------------
        elif risk == "LOW":

            self.state = "DRIVING"
            self.safe_frame_count = 0

            agent_action = "CONTINUE"

            reason = (
                "Scene assessed as low risk. "
                "Agent continues normal driving."
            )

        # -------------------------------------------------
        # UNKNOWN
        # -------------------------------------------------
        else:

            self.state = "CAUTION"
            self.safe_frame_count = 0

            agent_action = "CAUTION"

            reason = (
                "Decision confidence is insufficient. "
                "Agent defaults to cautious behavior."
            )

        self.last_action = agent_action

        return {
            "previous_state": previous_state,
            "state": self.state,
            "risk": risk,
            "input_action": action,
            "agent_action": agent_action,
            "reason": reason
        }

    # -------------------------------------------------
    # ENTER STOP
    # -------------------------------------------------
    def _enter_stop(self):

        if self.state != "STOPPED":
            self.stop_started_at = time.monotonic()

        self.state = "STOPPED"
        self.safe_frame_count = 0

    # -------------------------------------------------
    # HANDLE STOPPED STATE
    # -------------------------------------------------
    def _handle_stopped_state(
        self,
        risk,
        action
    ):

        # Risk returned while stopped.
        if (
            risk == "HIGH"
            or action == "STOP"
        ):
            self.safe_frame_count = 0

            return (
                "STOP",
                "Hazard remains active. "
                "Agent maintains STOP."
            )

        # Ensure minimum stop duration.
        if self.stop_started_at is not None:

            elapsed = (
                time.monotonic()
                - self.stop_started_at
            )

            if elapsed < self.stop_hold_seconds:

                return (
                    "WAIT",
                    "Immediate hazard cleared, but minimum "
                    "safety hold time has not elapsed."
                )

        # Minimum stop completed.
        self.state = "REASSESSING"
        self.safe_frame_count = 1

        return (
            "WAIT",
            "Initial hazard cleared. "
            "Agent begins scene reassessment."
        )

    # -------------------------------------------------
    # HANDLE REASSESSMENT
    # -------------------------------------------------
    def _handle_reassessment(
        self,
        risk,
        action
    ):

        # Hazard returned.
        if (
            risk == "HIGH"
            or action == "STOP"
        ):

            self._enter_stop()

            return (
                "STOP",
                "Hazard reappeared during reassessment. "
                "Agent returns to STOPPED state."
            )

        # Moderate risk still exists.
        if (
            risk == "MEDIUM"
            or action == "SLOW_DOWN"
        ):

            self.safe_frame_count = 0

            return (
                "WAIT",
                "Scene is not yet sufficiently safe. "
                "Agent continues reassessment."
            )

        # Completely safe frame.
        if (
            risk == "LOW"
            and action == "CONTINUE"
        ):

            self.safe_frame_count += 1

            if (
                self.safe_frame_count
                >= self.safe_frames_required
            ):

                self.state = "DRIVING"
                self.safe_frame_count = 0
                self.stop_started_at = None

                return (
                    "RESUME",
                    "Scene remained safe across multiple "
                    "reassessment frames. Agent resumes driving."
                )

            return (
                "WAIT",
                f"Safe scene observed "
                f"{self.safe_frame_count}/"
                f"{self.safe_frames_required} times. "
                f"Agent waits for confirmation."
            )

        # Unknown or caution result.
        self.safe_frame_count = 0

        return (
            "WAIT",
            "Scene safety is not fully confirmed. "
            "Agent continues reassessment."
        )