class DecisionFusion:
    """
    Fuses object-level risk assessment with VLM scene-level reasoning.

    Design principle:
    The VLM may increase caution, but it must not override a strong
    safety-critical decision produced by the deterministic perception
    pipeline.
    """

    RISK_PRIORITY = {
        "UNKNOWN": 0,
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3
    }

    ACTION_PRIORITY = {
        "UNKNOWN": 0,
        "CONTINUE": 1,
        "CAUTION": 2,
        "SLOW_DOWN": 3,
        "STOP": 4
    }

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------
    def fuse(self, assessed_objects, vlm_scene):

        object_decision = self._get_object_decision(
            assessed_objects
        )

        vlm_decision = self._get_vlm_decision(
            vlm_scene
        )

        final_risk = self._fuse_risk(
            object_decision["risk"],
            vlm_decision["risk"]
        )

        final_action = self._fuse_action(
            object_decision["action"],
            vlm_decision["action"]
        )

        reason = self._build_reason(
            object_decision,
            vlm_decision,
            final_risk,
            final_action
        )

        return {
            "object_risk": object_decision["risk"],
            "object_action": object_decision["action"],

            "vlm_risk": vlm_decision["risk"],
            "vlm_action": vlm_decision["action"],

            "final_risk": final_risk,
            "final_action": final_action,

            "reason": reason
        }

    # -------------------------------------------------
    # OBJECT-LEVEL DECISION
    # -------------------------------------------------
    def _get_object_decision(self, assessed_objects):

        if not assessed_objects:
            return {
                "risk": "LOW",
                "action": "CONTINUE"
            }

        highest_risk = "LOW"
        strongest_action = "CONTINUE"

        for obj in assessed_objects:

            risk = str(
                obj.get("risk_level", "UNKNOWN")
            ).upper()

            action = str(
                obj.get(
                    "recommended_action",
                    "UNKNOWN"
                )
            ).upper()

            if self.RISK_PRIORITY.get(
                risk, 0
            ) > self.RISK_PRIORITY.get(
                highest_risk, 0
            ):
                highest_risk = risk

            if self.ACTION_PRIORITY.get(
                action, 0
            ) > self.ACTION_PRIORITY.get(
                strongest_action, 0
            ):
                strongest_action = action

        return {
            "risk": highest_risk,
            "action": strongest_action
        }

    # -------------------------------------------------
    # VLM SCENE DECISION
    # -------------------------------------------------
    def _get_vlm_decision(self, vlm_scene):

        if not vlm_scene:
            return {
                "risk": "UNKNOWN",
                "action": "UNKNOWN"
            }

        risk = str(
            vlm_scene.get(
                "risk",
                "UNKNOWN"
            )
        ).upper()

        action = str(
            vlm_scene.get(
                "recommended_action",
                "UNKNOWN"
            )
        ).upper()

        if risk not in self.RISK_PRIORITY:
            risk = "UNKNOWN"

        if action not in self.ACTION_PRIORITY:
            action = "UNKNOWN"

        return {
            "risk": risk,
            "action": action
        }

    # -------------------------------------------------
    # RISK FUSION
    # -------------------------------------------------
    def _fuse_risk(
        self,
        object_risk,
        vlm_risk
    ):

        # VLM result unavailable:
        # trust deterministic perception.
        if vlm_risk == "UNKNOWN":
            return object_risk

        # Safety-first policy:
        # select the higher risk level.
        if self.RISK_PRIORITY[object_risk] >= \
                self.RISK_PRIORITY[vlm_risk]:

            return object_risk

        return vlm_risk

    # -------------------------------------------------
    # ACTION FUSION
    # -------------------------------------------------
    def _fuse_action(
        self,
        object_action,
        vlm_action
    ):

        # VLM result unavailable.
        if vlm_action == "UNKNOWN":
            return object_action

        # Select the more conservative action.
        if self.ACTION_PRIORITY[object_action] >= \
                self.ACTION_PRIORITY[vlm_action]:

            return object_action

        return vlm_action

    # -------------------------------------------------
    # EXPLAIN DECISION
    # -------------------------------------------------
    def _build_reason(
        self,
        object_decision,
        vlm_decision,
        final_risk,
        final_action
    ):

        object_risk = object_decision["risk"]
        object_action = object_decision["action"]

        vlm_risk = vlm_decision["risk"]
        vlm_action = vlm_decision["action"]

        if vlm_risk == "UNKNOWN":

            return (
                "VLM scene assessment is not available. "
                "Final decision is based on object-level "
                "perception and the deterministic safety pipeline."
            )

        if (
            object_risk == vlm_risk
            and object_action == vlm_action
        ):

            return (
                "Object-level perception and VLM scene reasoning "
                "agree on the driving decision."
            )

        if (
            self.RISK_PRIORITY.get(object_risk, 0)
            >
            self.RISK_PRIORITY.get(vlm_risk, 0)
        ):

            return (
                f"Object-level perception detected "
                f"{object_risk} risk with action "
                f"{object_action}, while the VLM assessed "
                f"the scene as {vlm_risk} risk with action "
                f"{vlm_action}. Safety-first fusion preserves "
                f"the more conservative decision: "
                f"{final_risk} / {final_action}."
            )

        if (
            self.RISK_PRIORITY.get(vlm_risk, 0)
            >
            self.RISK_PRIORITY.get(object_risk, 0)
        ):

            return (
                f"The VLM detected higher scene-level risk "
                f"({vlm_risk}) than object-level perception "
                f"({object_risk}). The system therefore escalates "
                f"the decision to {final_risk} / {final_action}."
            )

        return (
            f"Object-level and VLM decisions differ. "
            f"Safety-first fusion selects the more conservative "
            f"result: {final_risk} / {final_action}."
        )