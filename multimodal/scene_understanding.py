import json

from multimodal.prompts import SCENE_UNDERSTANDING_PROMPT


class SceneUnderstanding:

    def __init__(self, vlm):
        self.vlm = vlm

    def analyze(self, frame):

        response = self.vlm.analyze(
            frame,
            SCENE_UNDERSTANDING_PROMPT
        )

        return self._parse_response(response)

    @staticmethod
    def _parse_response(response):

        try:
            # Remove accidental markdown formatting
            cleaned_response = response.strip()

            cleaned_response = cleaned_response.replace(
                "```json", ""
            )

            cleaned_response = cleaned_response.replace(
                "```", ""
            )

            cleaned_response = cleaned_response.strip()

            result = json.loads(cleaned_response)

        except json.JSONDecodeError:

            return {
                "scene": response,
                "hazards": "UNKNOWN",
                "risk": "UNKNOWN",
                "recommended_action": "UNKNOWN",
                "parse_success": False
            }

        # Validate expected values
        valid_risks = {
            "LOW",
            "MEDIUM",
            "HIGH"
        }

        valid_actions = {
            "CONTINUE",
            "SLOW_DOWN",
            "STOP",
            "CAUTION"
        }

        risk = str(
            result.get("risk", "UNKNOWN")
        ).upper()

        action = str(
            result.get(
                "recommended_action",
                "UNKNOWN"
            )
        ).upper()

        if risk not in valid_risks:
            risk = "UNKNOWN"

        if action not in valid_actions:
            action = "UNKNOWN"

        return {
            "scene": result.get(
                "scene",
                ""
            ),

            "hazards": result.get(
                "hazards",
                "None"
            ),

            "risk": risk,

            "recommended_action": action,

            "parse_success": True
        }