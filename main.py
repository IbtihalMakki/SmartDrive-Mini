import cv2
import time
import threading

from perception.pipeline import PerceptionPipeline
from perception.extractor import PerceptionExtractor
from perception.visualizer import PerceptionVisualizer

from autonomy.localization import ObjectLocalizer
from autonomy.prediction import TrajectoryPredictor
from autonomy.sensor_fusion import SensorFusion
from autonomy.risk_assessment import RiskAssessor
from autonomy.decision_fusion import DecisionFusion
from autonomy.driving_agent import DrivingAgent

from multimodal.vlm import VisionLanguageModel
from multimodal.scene_understanding import SceneUnderstanding


# -------------------------------------------------
# ASYNCHRONOUS VLM WORKER
# -------------------------------------------------
class VLMWorker:

    def __init__(self, scene_understanding):

        self.scene_understanding = scene_understanding

        self.latest_scene = {
            "scene": "Waiting for VLM analysis...",
            "hazards": "UNKNOWN",
            "risk": "UNKNOWN",
            "recommended_action": "UNKNOWN",
            "parse_success": False
        }

        self.busy = False
        self.lock = threading.Lock()

    def analyze_async(self, frame):

        if self.busy:
            return

        self.busy = True
        frame_copy = frame.copy()

        thread = threading.Thread(
            target=self._analyze,
            args=(frame_copy,),
            daemon=True
        )

        thread.start()

    def _analyze(self, frame):

        try:

            print("\nRunning VLM scene analysis...")

            result = self.scene_understanding.analyze(
                frame
            )

            with self.lock:
                self.latest_scene = result

            print("\nVLM SCENE UNDERSTANDING")
            print("-" * 120)
            print(f"Scene: {result['scene']}")
            print(f"Hazards: {result['hazards']}")
            print(f"Risk: {result['risk']}")
            print(
                f"Action: "
                f"{result['recommended_action']}"
            )

        except Exception as error:

            print(
                f"\nVLM analysis failed: {error}"
            )

        finally:

            self.busy = False

    def get_latest_scene(self):

        with self.lock:
            return self.latest_scene.copy()


def main():

    # -------------------------------------------------
    # INITIALIZE PERCEPTION / AUTONOMY
    # -------------------------------------------------
    pipeline = PerceptionPipeline()

    localizer = ObjectLocalizer()

    predictor = TrajectoryPredictor(
        history_size=5,
        prediction_steps=5
    )

    sensor_fusion = SensorFusion()
    risk_assessor = RiskAssessor()
    decision_fusion = DecisionFusion()

    # -------------------------------------------------
    # INITIALIZE AGENTIC DECISION SYSTEM
    # -------------------------------------------------
    driving_agent = DrivingAgent(
        stop_hold_seconds=3.0,
        safe_frames_required=3
    )

    # Track agent changes so the console only prints
    # meaningful state/action transitions.
    last_agent_state = None
    last_agent_action = None

    # -------------------------------------------------
    # INITIALIZE MULTIMODAL VLM
    # -------------------------------------------------
    vlm = VisionLanguageModel()

    scene_understanding = SceneUnderstanding(
        vlm
    )

    vlm_worker = VLMWorker(
        scene_understanding
    )

    vlm_interval = 5.0
    last_vlm_time = 0.0

    # -------------------------------------------------
    # OPEN CAMERA
    # -------------------------------------------------
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open camera."
        )

    while True:

        success, frame = camera.read()

        if not success:
            break

        # -------------------------------------------------
        # 1. PERCEPTION
        # -------------------------------------------------
        results = pipeline.process(
            frame
        )

        # -------------------------------------------------
        # 2. STRUCTURED OBJECT EXTRACTION
        # -------------------------------------------------
        objects = PerceptionExtractor.extract_tracks(
            results["tracking"]
        )

        frame_height, frame_width = frame.shape[:2]

        # -------------------------------------------------
        # 3. LOCALIZATION + MOTION
        # -------------------------------------------------
        localized_objects = localizer.localize(
            objects,
            frame_width,
            frame_height
        )

        # -------------------------------------------------
        # 4. TRAJECTORY PREDICTION
        # -------------------------------------------------
        predicted_objects = predictor.predict(
            localized_objects
        )

        # -------------------------------------------------
        # 5. SENSOR FUSION
        # -------------------------------------------------
        fused_objects = sensor_fusion.fuse(
            predicted_objects,
            frame_height
        )

        # -------------------------------------------------
        # 6. OBJECT-LEVEL RISK ASSESSMENT
        # -------------------------------------------------
        assessed_objects = risk_assessor.assess(
            fused_objects
        )

        # -------------------------------------------------
        # 7. ASYNCHRONOUS VLM
        # -------------------------------------------------
        current_time = time.monotonic()

        if (
            current_time - last_vlm_time
            >= vlm_interval
        ):

            if not vlm_worker.busy:

                vlm_worker.analyze_async(
                    frame
                )

                last_vlm_time = current_time

        latest_scene = (
            vlm_worker.get_latest_scene()
        )

        # -------------------------------------------------
        # 8. MULTIMODAL DECISION FUSION
        # -------------------------------------------------
        final_decision = decision_fusion.fuse(
            assessed_objects,
            latest_scene
        )

        # -------------------------------------------------
        # 9. AGENTIC DECISION-MAKING
        # -------------------------------------------------
        agent_decision = driving_agent.step(
            final_decision
        )

        # -------------------------------------------------
        # 10. PRINT AGENT ONLY WHEN STATE/ACTION CHANGES
        # -------------------------------------------------
        current_agent_state = (
            agent_decision["state"]
        )

        current_agent_action = (
            agent_decision["agent_action"]
        )

        if (
            current_agent_state != last_agent_state
            or
            current_agent_action != last_agent_action
        ):

            print("\n" + "=" * 80)
            print("AGENT STATE CHANGE")
            print("=" * 80)

            print(
                f"State: "
                f"{agent_decision['previous_state']} "
                f"-> "
                f"{current_agent_state}"
            )

            print(
                f"Input: "
                f"{agent_decision['risk']} / "
                f"{agent_decision['input_action']}"
            )

            print(
                f"Agent Action: "
                f"{current_agent_action}"
            )

            print(
                f"Reason: "
                f"{agent_decision['reason']}"
            )

            last_agent_state = (
                current_agent_state
            )

            last_agent_action = (
                current_agent_action
            )

        # -------------------------------------------------
        # 11. VISUALIZATION
        # -------------------------------------------------
        annotated_frame = (
            PerceptionVisualizer.draw(
                frame,
                results
            )
        )

        # -------------------------------------------------
        # 12. DISPLAY VLM DECISION
        # -------------------------------------------------
        vlm_text = (
            f"VLM: "
            f"{latest_scene['risk']} | "
            f"{latest_scene['recommended_action']}"
        )

        cv2.putText(
            annotated_frame,
            vlm_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # -------------------------------------------------
        # 13. DISPLAY FUSED DECISION
        # -------------------------------------------------
        final_text = (
            f"FINAL: "
            f"{final_decision['final_risk']} | "
            f"{final_decision['final_action']}"
        )

        cv2.putText(
            annotated_frame,
            final_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # -------------------------------------------------
        # 14. DISPLAY AGENT STATE / ACTION
        # -------------------------------------------------
        agent_text = (
            f"AGENT: "
            f"{agent_decision['state']} | "
            f"{agent_decision['agent_action']}"
        )

        cv2.putText(
            annotated_frame,
            agent_text,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # -------------------------------------------------
        # 15. DISPLAY FRAME
        # -------------------------------------------------
        cv2.imshow(
            "SmartDrive-Mini | "
            "Multimodal Autonomous Driving Prototype",
            annotated_frame
        )

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()