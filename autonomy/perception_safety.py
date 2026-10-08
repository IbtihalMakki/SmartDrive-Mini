import time


class PerceptionSafetySupervisor:
    """
    Temporal safety layer for perception dropouts.

    Purpose:
    Prevent a recent HIGH / STOP obstacle from disappearing
    for one or two perception frames and immediately causing
    LOW / CONTINUE.

    IMPORTANT:
    - This layer operates on perception decisions only.
    - CARLA ground truth is NOT used for driving decisions.
    """

    def __init__(
        self,
        hazard_hold_seconds=1.0,
        safe_frames_required=3
    ):
        self.hazard_hold_seconds = hazard_hold_seconds
        self.safe_frames_required = safe_frames_required

        self.hazard_latched = False
        self.last_hazard_time = None

        self.safe_frame_count = 0

        self.last_hazard_reason = None

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.hazard_latched = False
        self.last_hazard_time = None
        self.safe_frame_count = 0
        self.last_hazard_reason = None

    # =========================================================
    # MAIN SAFETY CHECK
    # =========================================================

    def supervise(
        self,
        final_decision,
        assessed_objects
    ):
        """
        Input:
            final_decision:
                {
                    final_risk,
                    final_action,
                    decision_source,
                    ...
                }

            assessed_objects:
                physical objects after risk assessment

        Output:
            safe_decision
        """

        now = time.monotonic()

        original_risk = final_decision.get(
            "final_risk",
            "LOW"
        )

        original_action = final_decision.get(
            "final_action",
            "CONTINUE"
        )

        # =====================================================
        # DETERMINE WHETHER CURRENT FRAME HAS A REAL HAZARD
        # =====================================================

        high_risk_objects = [
            obj
            for obj in assessed_objects
            if (
                obj.get("risk_level") == "HIGH"
                or
                obj.get("recommended_action") == "STOP"
            )
        ]

        current_object_hazard = (
            len(high_risk_objects) > 0
        )

        # =====================================================
        # 1. LATCH NEW OBJECT HAZARD
        # =====================================================

        if current_object_hazard:

            self.hazard_latched = True
            self.last_hazard_time = now
            self.safe_frame_count = 0

            nearest_hazard = min(
                high_risk_objects,
                key=lambda obj: obj.get(
                    "distance_m",
                    float("inf")
                )
            )

            self.last_hazard_reason = (
                f"{nearest_hazard.get('class_name', 'object')} "
                f"@ "
                f"{nearest_hazard.get('distance_m', '?')}m"
            )

            result = dict(final_decision)

            result.update({
                "safety_override": False,
                "safety_latched": True,
                "safety_reason": (
                    "CURRENT_HIGH_RISK_OBJECT"
                ),
                "safe_frames": 0,
                "safe_frames_required":
                    self.safe_frames_required
            })

            return result

        # =====================================================
        # 2. NO ACTIVE LATCH
        # =====================================================

        if not self.hazard_latched:

            result = dict(final_decision)

            result.update({
                "safety_override": False,
                "safety_latched": False,
                "safety_reason": "NONE",
                "safe_frames": 0,
                "safe_frames_required":
                    self.safe_frames_required
            })

            return result

        # =====================================================
        # 3. RECENT HAZARD EXISTS
        # =====================================================

        elapsed_since_hazard = (
            now - self.last_hazard_time
            if self.last_hazard_time is not None
            else float("inf")
        )

        # =====================================================
        # 4. HOLD WINDOW
        # =====================================================

        if (
            elapsed_since_hazard
            < self.hazard_hold_seconds
        ):

            self.safe_frame_count = 0

            result = dict(final_decision)

            result.update({
                "final_risk": "HIGH",
                "final_action": "STOP",
                "decision_source":
                    "PERCEPTION_SAFETY",

                "safety_override": True,
                "safety_latched": True,

                "safety_reason":
                    "RECENT_HAZARD_HOLD",

                "safe_frames": 0,

                "safe_frames_required":
                    self.safe_frames_required,

                "last_hazard_reason":
                    self.last_hazard_reason,

                "hazard_age_seconds":
                    elapsed_since_hazard
            })

            return result

        # =====================================================
        # 5. HOLD WINDOW EXPIRED
        #
        # Now require consecutive safe evidence.
        # =====================================================

        frame_is_safe = (
            original_risk == "LOW"
            and original_action == "CONTINUE"
        )

        if frame_is_safe:

            self.safe_frame_count += 1

        else:

            self.safe_frame_count = 0

        # =====================================================
        # 6. RELEASE LATCH
        # =====================================================

        if (
            self.safe_frame_count
            >= self.safe_frames_required
        ):

            self.hazard_latched = False
            self.last_hazard_time = None
            self.safe_frame_count = 0

            previous_hazard = (
                self.last_hazard_reason
            )

            self.last_hazard_reason = None

            result = dict(final_decision)

            result.update({
                "safety_override": False,
                "safety_latched": False,

                "safety_reason":
                    "HAZARD_RELEASED",

                "released_hazard":
                    previous_hazard,

                "safe_frames":
                    self.safe_frames_required,

                "safe_frames_required":
                    self.safe_frames_required
            })

            return result

        # =====================================================
        # 7. STILL WAITING FOR SAFE CONFIRMATION
        # =====================================================

        result = dict(final_decision)

        result.update({
            "final_risk": "HIGH",
            "final_action": "STOP",
            "decision_source":
                "PERCEPTION_SAFETY",

            "safety_override": True,
            "safety_latched": True,

            "safety_reason":
                "WAITING_FOR_SAFE_CONFIRMATION",

            "safe_frames":
                self.safe_frame_count,

            "safe_frames_required":
                self.safe_frames_required,

            "last_hazard_reason":
                self.last_hazard_reason,

            "hazard_age_seconds":
                elapsed_since_hazard
        })

        return result