import math


class RiskAssessor:
    def __init__(self):
        pass

    def assess(self, objects):
        assessed_objects = []

        for obj in objects:
            distance = obj["distance_m"]
            region = obj["region"]
            motion = obj["motion"]

            # Missing metric depth must not look like a distant safe object.
            if not math.isfinite(distance) or obj.get("distance_valid") is False:
                assessed = obj.copy()
                assessed.update({
                    "risk_score": 4,
                    "risk_level": "HIGH",
                    "recommended_action": "STOP",
                    "risk_reason": "INVALID_DEPTH",
                })
                assessed_objects.append(assessed)
                continue

            # Default state
            risk_level = "LOW"
            action = "CONTINUE"
            risk_score = 0

            # -------------------------------------------------
            # 1. DISTANCE RISK
            # -------------------------------------------------
            if distance <= 5:
                risk_score += 2

            elif distance <= 10:
                risk_score += 1

            # -------------------------------------------------
            # 2. PATH / REGION RISK
            # Objects in the center are more relevant to
            # the ego vehicle's assumed forward path.
            # -------------------------------------------------
            if region == "CENTER":
                risk_score += 2

            # -------------------------------------------------
            # 3. MOTION RISK
            # A moving object gets a small additional weight.
            # -------------------------------------------------
            if motion not in ["STATIC", "UNKNOWN"]:
                risk_score += 1

            # -------------------------------------------------
            # 4. FINAL RISK DECISION
            # -------------------------------------------------
            if risk_score >= 4:
                risk_level = "HIGH"
                action = "STOP"

            elif risk_score >= 2:
                risk_level = "MEDIUM"
                action = "SLOW_DOWN"

            else:
                risk_level = "LOW"
                action = "CONTINUE"

            assessed = obj.copy()

            assessed.update({
                "risk_score": risk_score,
                "risk_level": risk_level,
                "recommended_action": action
            })

            assessed_objects.append(assessed)

        return assessed_objects
