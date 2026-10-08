import cv2
import numpy as np
import time
import json
from pathlib import Path
from simulation.rgbd_validation import validate_distance, DistanceValidationRun
from simulation.localization_validation import validate_localization, AXIS_TOLERANCE_M, POSITION_TOLERANCE_M
from autonomy.metric_localization import camera_intrinsics, localize_rgbd

from simulation.carla_client import CarlaClient
from simulation.rgbd import RGBDFrameBuffer

from perception.pipeline import PerceptionPipeline
from perception.extractor import PerceptionExtractor
from perception.visualizer import PerceptionVisualizer

from autonomy.localization import ObjectLocalizer
from autonomy.prediction import TrajectoryPredictor
from autonomy.sensor_fusion import SensorFusion
from autonomy.risk_assessment import RiskAssessor
from autonomy.decision_fusion import DecisionFusion
from autonomy.driving_agent import DrivingAgent
from autonomy.vehicle_controller import VehicleController
from autonomy.waypoint_navigator import WaypointNavigator
from autonomy.route_navigator import RouteNavigator
from evaluation.trajectory import evaluate as evaluate_trajectory
from autonomy.traffic_light_handler import TrafficLightHandler
from autonomy.perception_safety import PerceptionSafetySupervisor


# =============================================================
# TEST MODE
# =============================================================

# Available modes:
#
# "NPC_RECOVERY"
#     Isolated controlled-obstacle recovery test.
#     Traffic lights are still detected and logged,
#     but they do NOT control the vehicle.
#
# "PERCEPTION_DROPOUT"
#     Controlled temporal perception-dropout safety test.
#     A real HIGH/STOP obstacle is first detected and latched.
#     Then physical-object perception evidence is intentionally
#     suppressed while the CARLA NPC remains physically present.
#     Traffic-light control is disabled for test isolation.
#
# "TRAFFIC_LIGHT_TEST"
#     Deterministic CARLA semantic traffic-light validation.
#     Forces RED -> YELLOW -> GREEN and validates both the
#     semantic decision and vehicle/agent response.
#
# "NORMAL"
#     Normal safety-first behavior.
#     Traffic lights participate in the final decision.
#
# "RGBD_DISTANCE_TEST"
#     Synchronized RGB-D validation against actor-labelled semantic LiDAR.
#     Stops after ten consecutive passing frames or 200 attempts.
# "RGBD_3D_LOCALIZATION_TEST"
#     Metric optical XYZ surface localization against the same LiDAR reference.
# "ROUTE_NAVIGATION_TEST"
#     Global route planning and ordered waypoint following to a destination.
#
# IMPORTANT:
# NPC_RECOVERY, PERCEPTION_DROPOUT, and TRAFFIC_LIGHT_TEST are isolated test modes.

TEST_MODE = "NPC_RECOVERY"
ROUTE_START_INDEX = 0
ROUTE_DESTINATION_INDEX = None  # Auto-select reachable destination, or map spawn index.
ROUTE_TIMEOUT_SECONDS = 180.0
RGBD_VALIDATION_MODES = {"RGBD_DISTANCE_TEST", "RGBD_3D_LOCALIZATION_TEST"}


# =============================================================
# GLOBAL CAMERA FRAME
# =============================================================

rgbd_buffer = RGBDFrameBuffer()


def process_carla_image(image):
    rgbd_buffer.add("rgb", image)


def process_carla_depth(image):
    rgbd_buffer.add("depth", image)


# =============================================================
# CARLA GROUND TRUTH
# =============================================================

def get_carla_ground_truth(
    ego_vehicle,
    npc_vehicle
):
    """
    Get ego speed and true center-to-center distance
    to the controlled CARLA NPC.

    Ground truth is used for validation only.
    It is NOT used as a perception decision input.
    """

    velocity = ego_vehicle.get_velocity()

    speed_mps = (
        velocity.x ** 2
        + velocity.y ** 2
        + velocity.z ** 2
    ) ** 0.5

    speed_kmh = speed_mps * 3.6

    ego_location = ego_vehicle.get_location()
    npc_location = npc_vehicle.get_location()

    true_distance = ego_location.distance(
        npc_location
    )

    return {
        "speed_kmh": speed_kmh,
        "true_distance_m": true_distance
    }


# =============================================================
# NEAREST VISION VEHICLE
# =============================================================

def find_nearest_vision_vehicle(
    assessed_objects
):
    """
    Find nearest vehicle detected by vision.

    IMPORTANT:
    This does NOT prove that the detection is
    the controlled CARLA NPC.
    """

    vehicle_classes = {
        "car",
        "truck",
        "bus",
        "motorcycle"
    }

    vehicles = [
        obj
        for obj in assessed_objects
        if obj.get(
            "class_name",
            ""
        ).lower()
        in vehicle_classes
    ]

    if not vehicles:
        return None

    center_vehicles = [
        obj
        for obj in vehicles
        if obj.get("region") == "CENTER"
    ]

    candidates = (
        center_vehicles
        if center_vehicles
        else vehicles
    )

    return min(
        candidates,
        key=lambda obj: obj.get(
            "distance_m",
            float("inf")
        )
    )


# =============================================================
# MACHINE-READABLE BENCHMARK RESULTS
# =============================================================

def write_result_json(filename, payload):
    """Write one benchmark result under results/ and return its path."""
    result_dir = Path(__file__).parent / "results"
    result_dir.mkdir(exist_ok=True)
    path = result_dir / filename
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"BENCHMARK RESULT SAVED: {path}")
    return path


def compact_traffic_light_result(
    state,
    passed,
    traffic_light_status,
    final_decision,
    control,
    agent_decision,
    actor_verified,
):
    return {
        "status": "PASS" if passed else "NOT_PASSED",
        "test_mode": "TRAFFIC_LIGHT_TEST",
        "state": state,
        "semantic_actor_verified": bool(actor_verified),
        "detected": bool(traffic_light_status.get("detected", False)),
        "reported_state": traffic_light_status.get("state"),
        "traffic_risk": traffic_light_status.get("risk"),
        "traffic_action": traffic_light_status.get("action"),
        "final_risk": final_decision.get("final_risk"),
        "final_action": final_decision.get("final_action"),
        "decision_source": final_decision.get("decision_source"),
        "agent_state": agent_decision.get("state"),
        "agent_action": agent_decision.get("agent_action"),
        "control": {
            "throttle": float(control.throttle),
            "brake": float(control.brake),
            "steer": float(control.steer),
        },
        "distance_m": traffic_light_status.get("distance_m"),
        "validation": (
            "CARLA semantic traffic-light state; "
            "not visual color classification"
        ),
    }


# =============================================================
# MAIN
# =============================================================

def main():

    global rgbd_buffer
    rgbd_buffer = RGBDFrameBuffer(validation=TEST_MODE in RGBD_VALIDATION_MODES)
    distance_validation = DistanceValidationRun()

    trajectory_records = []
    trajectory_log = None
    route_navigator = None
    route_outcome = "NOT_STARTED"

    # Keep the camera window alive after a successful route test
    # so screenshots / recordings can be captured before cleanup.
    route_test_finished = False

    simulator = CarlaClient()

    print(
        "\nInitializing SmartDrive "
        "perception and autonomy pipeline..."
    )

    # =========================================================
    # PERCEPTION
    # =========================================================

    pipeline = PerceptionPipeline()

    # =========================================================
    # AUTONOMY MODULES
    # =========================================================

    localizer = ObjectLocalizer()

    predictor = TrajectoryPredictor(
        history_size=5,
        prediction_steps=5
    )

    sensor_fusion = SensorFusion()

    risk_assessor = RiskAssessor()

    decision_fusion = DecisionFusion()

    perception_safety = (
        PerceptionSafetySupervisor(
            hazard_hold_seconds=1.0,
            safe_frames_required=3
        )
    )

    driving_agent = DrivingAgent(
        stop_hold_seconds=3.0,
        safe_frames_required=3
    )

    # =========================================================
    # RECOVERY TEST CONFIGURATION
    # =========================================================

    recovery_state = "APPROACHING"

    stop_started_at = None

    npc_cleared = False

    resume_started_at = None

    recovery_passed_at = None

    new_hazard_after_recovery = False

    new_hazard_reason = None

    NPC_CLEAR_DELAY = 5.0

    RECOVERY_DRIVE_SECONDS_REQUIRED = 3.0

    MIN_RECOVERY_SPEED_KMH = 0.5

    # =========================================================
    # PERCEPTION DROPOUT TEST CONFIGURATION
    # =========================================================

    dropout_state = "WAITING_FOR_HAZARD"
    dropout_started_at = None

    dropout_latch_verified = False
    dropout_override_verified = False
    dropout_stop_verified = False
    dropout_release_verified = False

    DROPOUT_DURATION_SECONDS = 1.5

    # After perception is restored, deterministically remove the
    # controlled hazard actor from the CARLA world. This prevents the
    # controlled NPC from remaining a MEDIUM/SLOW_DOWN perception target
    # at long range, while leaving the safety supervisor untouched.
    dropout_npc_cleared = False

    # =========================================================
    # DETERMINISTIC TRAFFIC-LIGHT TEST CONFIGURATION
    # =========================================================

    traffic_test_state = "SETUP"
    traffic_test_state_started_at = None

    red_test_passed = False
    yellow_test_passed = False
    green_test_passed = False
    traffic_result_written = {
        "RED": False,
        "YELLOW": False,
        "GREEN": False,
    }

    dropout_result_written = False
    recovery_result_written = False

    TRAFFIC_STATE_HOLD_SECONDS = 2.0
    TRAFFIC_LIGHT_START_DISTANCE = 18.0

    try:

        # =====================================================
        # CARLA SETUP
        # =====================================================

        if TEST_MODE == "TRAFFIC_LIGHT_TEST":

            simulator.spawn_ego_for_traffic_light_test(
                distance_before_light=(
                    TRAFFIC_LIGHT_START_DISTANCE
                )
            )

            # No controlled NPC is spawned in the isolated
            # traffic-light test. This prevents obstacle risk
            # from masking the traffic-light decision.

        elif TEST_MODE == "ROUTE_NAVIGATION_TEST":
            simulator.spawn_ego_vehicle(spawn_index=ROUTE_START_INDEX)

        else:

            simulator.spawn_ego_vehicle(spawn_index=0 if TEST_MODE == "TRAJECTORY_PREDICTION_TEST" else None)

            simulator.spawn_npc_ahead(
                distance=20.0
            )

        simulator.spawn_rgb_camera(
            callback=process_carla_image,
            width=1280,
            height=720,
            fov=90,
            pinhole=TEST_MODE in RGBD_VALIDATION_MODES
        )

        simulator.spawn_depth_camera(callback=process_carla_depth)
        if TEST_MODE in RGBD_VALIDATION_MODES:
            simulator.start_distance_validation(
                callback=lambda measurement: rgbd_buffer.add("reference", measurement)
            )

        # =====================================================
        # VEHICLE
        # =====================================================

        ego_vehicle = simulator.ego_vehicle

        carla_map = simulator.world.get_map()

        # =====================================================
        # VEHICLE CONTROLLER
        # =====================================================

        vehicle_controller = VehicleController(
            ego_vehicle
        )

        # =====================================================
        # WAYPOINT NAVIGATION
        # =====================================================

        navigator = WaypointNavigator(
            carla_map,
            ego_vehicle
        )

        if TEST_MODE == "ROUTE_NAVIGATION_TEST":
            route_navigator = RouteNavigator.plan(carla_map, ego_vehicle, ROUTE_DESTINATION_INDEX)
            navigator = route_navigator
            route_outcome = "RUNNING"
            print(f"ROUTE GENERATED: {len(navigator.points)} waypoints, "
                  f"{navigator.route_length_m:.1f} m, destination index={navigator.destination_index}")

        # =====================================================
        # TRAFFIC LIGHT HANDLER
        # =====================================================

        traffic_light_handler = (
            TrafficLightHandler(
                ego_vehicle,
                simulator.world,
                max_detection_distance=35.0
            )
        )

        # =====================================================
        # STARTUP INFO
        # =====================================================

        print(
            "\nCARLA -> SmartDrive Agent "
            "integration is running."
        )

        print(
            "Closed-loop control ENABLED."
        )

        print(
            "Controlled NPC test ENABLED."
        )

        print(
            "Traffic-light semantics ENABLED."
        )

        print(
            "Perception Dropout Safety ENABLED."
        )

        print(
            "Time-based Recovery Validator ENABLED."
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "TEST CONFIGURATION"
        )

        print(
            f"Mode: {TEST_MODE}"
        )

        if TEST_MODE in {"NPC_RECOVERY", "PERCEPTION_DROPOUT"}:

            print(
                "Traffic-Light Control: "
                "DISABLED FOR ISOLATED TEST"
            )

            print(
                "Traffic lights will still be "
                "detected and logged."
            )

        elif TEST_MODE == "TRAFFIC_LIGHT_TEST":

            print(
                "Traffic-Light Control: "
                "ENABLED FOR DETERMINISTIC TEST"
            )

            print(
                "Sequence: RED -> YELLOW -> GREEN"
            )

        else:

            print(
                "Traffic-Light Control: ENABLED"
            )

        print(
            "=" * 70
        )

        print(
            "\nPerception safety configuration:"
        )

        print(
            "Hazard hold: 1.0 second"
        )

        print(
            "Safe confirmation: 3 frames"
        )

        if TEST_MODE == "NPC_RECOVERY":

            print(
                "\nExpected recovery sequence:"
            )

            print(
                "APPROACHING"
                " -> STOP_VERIFIED"
                " -> WAITING_TO_CLEAR"
                " -> WAITING_FOR_RESUME"
                " -> RESUME_CANDIDATE"
                " -> RECOVERY_PASSED"
            )

            print(
                "\nRecovery requires "
                f"{RECOVERY_DRIVE_SECONDS_REQUIRED:.1f} "
                "continuous seconds of autonomous driving."
            )

        elif TEST_MODE == "PERCEPTION_DROPOUT":

            print(
                "\nExpected dropout sequence:"
            )

            print(
                "WAITING_FOR_HAZARD"
                " -> DROPOUT_ACTIVE"
                " -> PERCEPTION_RESTORED"
                " -> DROPOUT_TEST_PASSED"
            )

            print(
                f"\nInjected dropout duration: "
                f"{DROPOUT_DURATION_SECONDS:.1f} seconds"
            )

        elif TEST_MODE == "TRAFFIC_LIGHT_TEST":

            print(
                "\nExpected traffic-light sequence:"
            )

            print(
                "RED -> STOP -> "
                "YELLOW -> SLOW_DOWN -> "
                "GREEN -> CONTINUE/RESUME -> "
                "TRAFFIC_LIGHT_TEST_PASSED"
            )

        print(
            "\nPress Q in the camera window "
            "to stop."
        )

        # =====================================================
        # INITIALIZE DETERMINISTIC TRAFFIC-LIGHT TEST
        # =====================================================

        if TEST_MODE == "TRAFFIC_LIGHT_TEST":

            simulator.set_test_traffic_light_state(
                "RED"
            )

            traffic_test_state = "RED"
            traffic_test_state_started_at = (
                time.monotonic()
            )

            # Give CARLA a short moment to propagate the
            # forced semantic state to the vehicle API.
            time.sleep(0.25)

        # =====================================================
        # MAIN LOOP
        # =====================================================

        if TEST_MODE == "TRAJECTORY_PREDICTION_TEST":
            output_dir = Path(__file__).parent / "results"
            output_dir.mkdir(exist_ok=True)
            trajectory_log = (output_dir / "trajectory_observations.jsonl").open("w", encoding="utf-8")
            (output_dir / "trajectory_metadata.json").write_text(json.dumps({
                "map": carla_map.name, "start_index": 0,
                "predictor": "image-space mean velocity with static gating and direction correction",
                "history_size": 5, "prediction_steps": 5,
                "scenario": "stationary NPC ahead; ego follows lane", "target_frames": 200,
            }, indent=2), encoding="utf-8")
        route_started_at = time.monotonic()
        last_rgbd_received = time.monotonic()
        rgbd_waiting = False

        while True:

            # =================================================
            # GET CAMERA FRAME
            # =================================================

            if TEST_MODE == "TRAJECTORY_PREDICTION_TEST" and time.monotonic() - route_started_at > 180:
                vehicle_controller.apply_action("STOP")
                print("TRAJECTORY COLLECTION TIMEOUT: evaluate available observations")
                break
            if (
                route_navigator is not None
                and not route_test_finished
                and time.monotonic() - route_started_at > ROUTE_TIMEOUT_SECONDS
            ):
                route_outcome = "TIMEOUT"
                vehicle_controller.apply_action("STOP")
                print("ROUTE_NAVIGATION_TEST_FAILED: timeout")
                break
            if TEST_MODE in RGBD_VALIDATION_MODES:
                simulator.world.tick()
                deadline = time.monotonic() + 2.0
                pair = rgbd_buffer.pop()
                while pair is None and time.monotonic() < deadline:
                    time.sleep(0.005)
                    pair = rgbd_buffer.pop()
                if pair is None:
                    vehicle_controller.apply_action("STOP")
                    distance_validation.record(simulator.world.get_snapshot().frame,
                                               {"status": "FAIL", "reason": "SENSOR_TIMEOUT"})
                    raise RuntimeError("RGB-D validation sensor synchronization timed out")
            else:
                pair = rgbd_buffer.pop()
            if pair is None:
                if time.monotonic() - last_rgbd_received > 2.0:
                    vehicle_controller.apply_action("STOP")
                    if not rgbd_waiting:
                        print("RGB-D WAIT: no fresh matched frames; vehicle held stopped")
                        rgbd_waiting = True
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.005)
                continue

            rgbd_frame_id, rgbd_timestamp, frame, depth_m = pair[:4]
            last_rgbd_received = time.monotonic()
            rgbd_waiting = False

            frame_height, frame_width = (
                frame.shape[:2]
            )

            # =================================================
            # 1. PERCEPTION
            # =================================================

            results = pipeline.process(
                frame
            )

            # =================================================
            # 2. TRACK EXTRACTION
            # =================================================

            objects = (
                PerceptionExtractor
                .extract_tracks(
                    results["tracking"]
                )
            )

            # =================================================
            # 3. LOCALIZATION
            # =================================================

            localized_objects = (
                localizer.localize(
                    objects,
                    frame_width,
                    frame_height
                )
            )

            # =================================================
            # 4. TRAJECTORY PREDICTION
            # =================================================

            predicted_objects = (
                predictor.predict(
                    localized_objects
                )
            )

            if TEST_MODE == "TRAJECTORY_PREDICTION_TEST":
                record = dict(scenario="controlled_npc_approach", frame=rgbd_frame_id,
                              timestamp=rgbd_timestamp, objects=[
                                  {key: obj[key] for key in ("track_id", "class_name", "center", "predicted_centers")}
                                  for obj in predicted_objects
                                  if obj["class_name"].lower() in {"car", "truck", "bus", "motorcycle", "person", "bicycle"}
                              ])
                trajectory_records.append(record)
                trajectory_log.write(json.dumps(record) + "\n")
                trajectory_log.flush()
                if len(trajectory_records) >= 200:
                    vehicle_controller.apply_action("STOP")
                    print("TRAJECTORY COLLECTION COMPLETE: 200 frames")
                    break

            # =================================================
            # 5. SENSOR FUSION
            # =================================================

            fused_objects = (
                sensor_fusion.fuse_rgbd(
                    predicted_objects,
                    depth_m,
                    rgbd_frame_id
                )
            )

            if TEST_MODE in RGBD_VALIDATION_MODES:
                intrinsics = camera_intrinsics(
                    frame_width, frame_height, float(simulator.rgb_camera.attributes["fov"])
                )
                fused_objects = localize_rgbd(fused_objects, depth_m, rgbd_frame_id, intrinsics)
                validator = (validate_localization if TEST_MODE == "RGBD_3D_LOCALIZATION_TEST"
                             else validate_distance)
                validation = validator(
                    fused_objects, pair[4], simulator.npc_vehicle.id,
                    frame_width, frame_height,
                    float(simulator.rgb_camera.attributes["fov"]),
                )
                distance_validation.record(rgbd_frame_id, validation)
                print("RGB-D 3D LOCALIZATION VALIDATION" if TEST_MODE == "RGBD_3D_LOCALIZATION_TEST"
                      else "RGB-D SENSOR FUSION VALIDATION")
                print(f"Controlled NPC Association: {validation['association']}")
                print(f"Frame: {rgbd_frame_id}; NPC actor: {simulator.npc_vehicle.id}; "
                      f"track: {validation.get('track_id', 'NONE')}")
                if "predicted_xyz_m" in validation:
                    print(f"Predicted X/Y/Z (m): {validation['predicted_xyz_m']}")
                    print(f"Reference X/Y/Z (m): {validation['reference_xyz_m']}")
                    print(f"Axis absolute errors (m): {validation['axis_absolute_error_m']}")
                    print(f"Euclidean position error: {validation['position_error_m']:.3f} m")
                    print(f"Axis tolerances (m): {AXIS_TOLERANCE_M}; norm: {POSITION_TOLERANCE_M}")
                if "error_m" in validation:
                    print(f"RGB-D Distance: {validation['depth_m']:.3f} m")
                    print(f"Reference Surface Depth: {validation['reference_m']:.3f} m")
                    print(f"Absolute Error: {validation['error_m']:.3f} m")
                    print(f"Tolerance: {validation['tolerance_m']:.3f} m")
                print(f"VALIDATION: {validation['status']} ({validation['reason']})")
                print(f"Consecutive passing frames: {distance_validation.streak}/10")
                if distance_validation.passed:
                    print(f"{TEST_MODE}_PASSED")
                    vehicle_controller.apply_action("STOP")
                    break
                if len(distance_validation.results) >= 200:
                    print(f"{TEST_MODE}_FAILED: no ten-frame passing sequence")
                    vehicle_controller.apply_action("STOP")
                    break

            valid_depth_count = sum(obj["distance_valid"] for obj in fused_objects)
            print(
                f"RGB-D: frame={rgbd_frame_id}, timestamp={rgbd_timestamp:.3f}, "
                f"matched=True, valid_objects={valid_depth_count}/{len(fused_objects)}"
            )
            for obj in fused_objects:
                print(
                    f"  track={obj.get('track_id')} depth={obj['distance_m']:.2f} m "
                    f"source={obj['distance_source']} samples={obj['depth_sample_count']}"
                )

            # =================================================
            # 6. SEPARATE TRAFFIC LIGHTS
            # =================================================

            traffic_light_detections = []

            physical_objects = []

            for obj in fused_objects:

                class_name = (
                    obj.get(
                        "class_name",
                        ""
                    ).lower()
                )

                if class_name == "traffic light":

                    traffic_light_detections.append(
                        obj
                    )

                else:

                    physical_objects.append(
                        obj
                    )

            # =================================================
            # 7. PHYSICAL OBSTACLE RISK
            # =================================================

            assessed_objects = (
                risk_assessor.assess(
                    physical_objects
                )
            )

            # =================================================
            # CONTROLLED PERCEPTION DROPOUT INJECTION
            # =================================================

            if TEST_MODE == "PERCEPTION_DROPOUT":

                dropout_now = time.monotonic()

                has_high_hazard = any(
                    obj.get("risk_level") == "HIGH"
                    or obj.get("recommended_action") == "STOP"
                    for obj in assessed_objects
                )

                # Start the dropout only AFTER at least one real
                # HIGH/STOP frame has already reached the safety
                # supervisor. This guarantees a genuine latch exists.
                if (
                    dropout_state == "WAITING_FOR_HAZARD"
                    and perception_safety.hazard_latched
                ):

                    dropout_state = "DROPOUT_ACTIVE"
                    dropout_started_at = dropout_now

                    print("\n" + "=" * 70)
                    print("CONTROLLED PERCEPTION DROPOUT INJECTED")
                    print("A real HIGH/STOP hazard was previously latched.")
                    print("NPC remains physically present in CARLA.")
                    print(
                        "Physical-object perception evidence is now "
                        "intentionally suppressed."
                    )
                    print("=" * 70)

                if dropout_state == "DROPOUT_ACTIVE":

                    dropout_elapsed = (
                        dropout_now - dropout_started_at
                    )

                    if (
                        dropout_elapsed
                        < DROPOUT_DURATION_SECONDS
                    ):

                        # IMPORTANT:
                        # The CARLA NPC is not moved or removed.
                        # Only the physical-object perception evidence
                        # delivered downstream is suppressed.
                        assessed_objects = []

                    else:

                        dropout_state = "PERCEPTION_RESTORED"

                        print("\n" + "=" * 70)
                        print("PERCEPTION RESTORED")
                        print(
                            f"Controlled dropout duration: "
                            f"{dropout_elapsed:.2f} s"
                        )
                        print(
                            "Normal physical-object perception "
                            "evidence is active again."
                        )

                        # Deterministically remove the controlled hazard
                        # actor from the CARLA world. We deliberately do NOT
                        # reset/modify PerceptionSafetySupervisor: the latch
                        # must release by its normal hold + safe-frame logic.
                        if not dropout_npc_cleared:
                            npc_actor = simulator.npc_vehicle

                            if npc_actor is not None:
                                try:
                                    npc_actor.destroy()
                                    simulator.npc_vehicle = None
                                    dropout_npc_cleared = True
                                except RuntimeError as exc:
                                    print(
                                        "WARNING: controlled NPC could not "
                                        f"be destroyed: {exc}"
                                    )
                            else:
                                # If it was already removed, treat the
                                # controlled hazard as cleared.
                                dropout_npc_cleared = True

                            if dropout_npc_cleared:
                                # Remove stale tracker/risk evidence from this
                                # transition frame only. Subsequent frames use
                                # normal perception again; any real remaining
                                # hazard can still prevent release.
                                assessed_objects = []

                                print(
                                    "CONTROLLED NPC REMOVED AFTER DROPOUT"
                                )
                                print(
                                    "Safety supervisor remains latched; "
                                    "waiting for normal safe-frame release."
                                )
                            else:
                                print(
                                    "WARNING: controlled NPC was not cleared; "
                                    "release will not be accepted as PASS."
                                )

                        print("=" * 70)

            # =================================================
            # 8. CARLA TRAFFIC-LIGHT STATE
            # =================================================

            if TEST_MODE == "TRAFFIC_LIGHT_TEST":
                # CARLA can change which traffic-light actor is associated
                # with ego after simulation ticks. Re-apply the requested
                # state to that exact actor before reading semantic status.
                simulator.refresh_test_traffic_light_state()

            traffic_light_status = (
                traffic_light_handler
                .get_status()
            )

            if TEST_MODE == "TRAFFIC_LIGHT_TEST":
                traffic_test_actor_verified = (
                    simulator.verify_test_traffic_light_state(
                        traffic_light_status
                    )
                )

            # =================================================
            # 9. OBJECT DECISION
            # =================================================

            vlm_scene = {
                "risk": "UNKNOWN",
                "recommended_action": "UNKNOWN"
            }

            object_decision = (
                decision_fusion.fuse(
                    assessed_objects,
                    vlm_scene
                )
            )

            object_risk = (
                object_decision[
                    "final_risk"
                ]
            )

            object_action = (
                object_decision[
                    "final_action"
                ]
            )

            traffic_risk = (
                traffic_light_status[
                    "risk"
                ]
            )

            traffic_action = (
                traffic_light_status[
                    "action"
                ]
            )

            # =================================================
            # 10. TEST-MODE DECISION POLICY
            # =================================================

            risk_priority = {
                "UNKNOWN": 0,
                "LOW": 1,
                "MEDIUM": 2,
                "HIGH": 3
            }

            # -------------------------------------------------
            # NPC RECOVERY TEST
            # -------------------------------------------------

            if TEST_MODE in {"NPC_RECOVERY", "PERCEPTION_DROPOUT"}:

                # Traffic-light telemetry remains active.
                #
                # Traffic-light control is intentionally
                # excluded ONLY from this isolated test.
                #
                # Never use this mode as normal driving mode.

                final_decision = {

                    "object_risk":
                        object_risk,

                    "object_action":
                        object_action,

                    "vlm_risk":
                        "UNKNOWN",

                    "vlm_action":
                        "UNKNOWN",

                    "traffic_light_risk":
                        traffic_risk,

                    "traffic_light_action":
                        traffic_action,

                    "final_risk":
                        object_risk,

                    "final_action":
                        object_action,

                    "decision_source":
                        "OBJECTS",

                    "test_mode":
                        TEST_MODE,

                    "traffic_light_control_enabled":
                        False
                }

            # -------------------------------------------------
            # TRAFFIC-LIGHT TEST / NORMAL MODE
            # -------------------------------------------------

            else:

                if (
                    risk_priority.get(
                        traffic_risk,
                        0
                    )
                    >
                    risk_priority.get(
                        object_risk,
                        0
                    )
                ):

                    final_decision = {

                        "object_risk":
                            object_risk,

                        "object_action":
                            object_action,

                        "vlm_risk":
                            "UNKNOWN",

                        "vlm_action":
                            "UNKNOWN",

                        "traffic_light_risk":
                            traffic_risk,

                        "traffic_light_action":
                            traffic_action,

                        "final_risk":
                            traffic_risk,

                        "final_action":
                            traffic_action,

                        "decision_source":
                            "TRAFFIC_LIGHT",

                        "test_mode":
                            TEST_MODE,

                        "traffic_light_control_enabled":
                            True
                    }

                else:

                    final_decision = {

                        "object_risk":
                            object_risk,

                        "object_action":
                            object_action,

                        "vlm_risk":
                            "UNKNOWN",

                        "vlm_action":
                            "UNKNOWN",

                        "traffic_light_risk":
                            traffic_risk,

                        "traffic_light_action":
                            traffic_action,

                        "final_risk":
                            object_risk,

                        "final_action":
                            object_action,

                        "decision_source":
                            "OBJECTS",

                        "test_mode":
                            TEST_MODE,

                        "traffic_light_control_enabled":
                            True
                    }

            # =================================================
            # 11. PERCEPTION DROPOUT SAFETY
            # =================================================

            final_decision = (
                perception_safety.supervise(
                    final_decision,
                    assessed_objects
                )
            )

            # =================================================
            # PERCEPTION DROPOUT TEST VALIDATION
            # =================================================

            if TEST_MODE == "PERCEPTION_DROPOUT":

                if dropout_state == "DROPOUT_ACTIVE":

                    if final_decision.get(
                        "safety_latched",
                        False
                    ):
                        dropout_latch_verified = True

                    if final_decision.get(
                        "safety_override",
                        False
                    ):
                        dropout_override_verified = True

                    if (
                        final_decision.get("final_risk")
                        == "HIGH"
                        and
                        final_decision.get("final_action")
                        == "STOP"
                        and
                        final_decision.get("decision_source")
                        == "PERCEPTION_SAFETY"
                    ):
                        dropout_stop_verified = True

                if (
                    dropout_state == "PERCEPTION_RESTORED"
                    and
                    dropout_latch_verified
                    and
                    dropout_override_verified
                    and
                    dropout_stop_verified
                    and
                    dropout_npc_cleared
                    and
                    not final_decision.get(
                        "safety_latched",
                        False
                    )
                ):

                    dropout_release_verified = True
                    dropout_state = "DROPOUT_TEST_PASSED"

                    print("\n" + "=" * 70)
                    print("PERCEPTION DROPOUT TEST PASSED")
                    print("=" * 70)
                    print("Hazard latch during dropout: PASS")
                    print("Safety override during dropout: PASS")
                    print("HIGH / STOP maintained:        PASS")
                    print("Perception restored:           PASS")
                    print("Safety latch released:         PASS")
                    print("=" * 70)

                    if not dropout_result_written:
                        write_result_json(
                            "perception_dropout_result.json",
                            {
                                "status": "PASS",
                                "test_mode": TEST_MODE,
                                "dropout_duration_seconds": DROPOUT_DURATION_SECONDS,
                                "hazard_latch_verified": dropout_latch_verified,
                                "safety_override_verified": dropout_override_verified,
                                "high_stop_maintained": dropout_stop_verified,
                                "perception_restored": True,
                                "controlled_npc_cleared": bool(dropout_npc_cleared),
                                "controlled_npc_clear_method": "DESTROY_ACTOR_AFTER_RESTORE",
                                "safety_latch_released": dropout_release_verified,
                                "validation_scope": (
                                    "Temporal perception-dropout safety / "
                                    "hazard persistence after a real HIGH/STOP latch"
                                ),
                            },
                        )
                        dropout_result_written = True

            # =================================================
            # 12. STATEFUL DRIVING AGENT
            # =================================================

            agent_decision = (
                driving_agent.step(
                    final_decision
                )
            )

            # =================================================
            # 13. WAYPOINT STEERING
            # =================================================

            steer = navigator.get_steer()
            if route_navigator is not None:
                print(f"ROUTE: reached={navigator.index}/{len(navigator.points)}, "
                      f"destination distance={navigator.history[-1]['destination_distance_m']:.2f} m")
                if navigator.exhausted_without_arrival:
                    vehicle_controller.apply_action("STOP")
                    route_outcome = "ROUTE_EXHAUSTED_WITHOUT_ARRIVAL"
                    print("ROUTE_NAVIGATION_TEST_FAILED: route exhausted without arrival")
                    break
                if navigator.reached:
                    route_outcome = "PASS"

                    if not route_test_finished:
                        route_test_finished = True
                        vehicle_controller.apply_action("STOP", steer=0.0)

                        print("\n" + "=" * 70)
                        print("ROUTE_NAVIGATION_TEST_PASSED")
                        print(
                            f"Destination reached: "
                            f"{navigator.history[-1]['destination_distance_m']:.2f} m"
                        )
                        print("Vehicle stopped safely.")
                        print("Camera window will remain open.")
                        print("Press Q in the camera window to exit.")
                        print("=" * 70)

            # =================================================
            # 14. VEHICLE CONTROL
            # =================================================

            if (
                TEST_MODE == "ROUTE_NAVIGATION_TEST"
                and route_test_finished
            ):
                # Route test already passed. Keep ego stationary while
                # visualization continues until the user presses Q.
                control = vehicle_controller.apply_action(
                    "STOP",
                    steer=0.0
                )
            else:
                control = (
                    vehicle_controller
                    .apply_action(
                        agent_decision[
                            "agent_action"
                        ],
                        steer=steer
                    )
                )

            # =================================================
            # DETERMINISTIC TRAFFIC-LIGHT TEST VALIDATION
            # =================================================

            if TEST_MODE == "TRAFFIC_LIGHT_TEST":

                traffic_now = time.monotonic()

                state_age = (
                    traffic_now
                    - traffic_test_state_started_at
                    if traffic_test_state_started_at
                    is not None
                    else 0.0
                )

                if traffic_test_state == "RED":

                    red_semantics_ok = (
                        traffic_test_actor_verified
                        and
                        traffic_light_status["detected"]
                        and
                        traffic_light_status["state"] == "RED"
                        and
                        traffic_light_status["risk"] == "HIGH"
                        and
                        traffic_light_status["action"] == "STOP"
                        and
                        final_decision["final_risk"] == "HIGH"
                        and
                        final_decision["final_action"] == "STOP"
                        and
                        final_decision["decision_source"]
                        == "TRAFFIC_LIGHT"
                    )

                    red_control_ok = (
                        control.throttle <= 0.01
                        and
                        control.brake >= 0.5
                    )

                    if (
                        red_semantics_ok
                        and red_control_ok
                        and state_age
                        >= TRAFFIC_STATE_HOLD_SECONDS
                    ):

                        red_test_passed = True

                        print("\n" + "=" * 70)
                        print("RED_TEST_PASSED")
                        print("RED -> HIGH / STOP")
                        print(
                            f"Control: T={control.throttle:.2f} "
                            f"B={control.brake:.2f}"
                        )
                        print("=" * 70)

                        if not traffic_result_written["RED"]:
                            write_result_json(
                                "traffic_light_red_result.json",
                                compact_traffic_light_result(
                                    "RED", True, traffic_light_status,
                                    final_decision, control, agent_decision,
                                    traffic_test_actor_verified,
                                ),
                            )
                            traffic_result_written["RED"] = True

                        simulator.set_test_traffic_light_state(
                            "YELLOW"
                        )

                        traffic_test_state = "YELLOW"
                        traffic_test_state_started_at = (
                            traffic_now
                        )

                elif traffic_test_state == "YELLOW":

                    yellow_semantics_ok = (
                        traffic_test_actor_verified
                        and
                        traffic_light_status["detected"]
                        and
                        traffic_light_status["state"]
                        == "YELLOW"
                        and
                        traffic_light_status["risk"]
                        == "MEDIUM"
                        and
                        traffic_light_status["action"]
                        == "SLOW_DOWN"
                        and
                        final_decision["final_risk"]
                        == "MEDIUM"
                        and
                        final_decision["final_action"]
                        == "SLOW_DOWN"
                        and
                        final_decision["decision_source"]
                        == "TRAFFIC_LIGHT"
                    )

                    if (
                        yellow_semantics_ok
                        and state_age
                        >= TRAFFIC_STATE_HOLD_SECONDS
                    ):

                        yellow_test_passed = True

                        print("\n" + "=" * 70)
                        print("YELLOW_TEST_PASSED")
                        print(
                            "YELLOW -> MEDIUM / SLOW_DOWN"
                        )
                        print("=" * 70)

                        if not traffic_result_written["YELLOW"]:
                            write_result_json(
                                "traffic_light_yellow_result.json",
                                compact_traffic_light_result(
                                    "YELLOW", True, traffic_light_status,
                                    final_decision, control, agent_decision,
                                    traffic_test_actor_verified,
                                ),
                            )
                            traffic_result_written["YELLOW"] = True

                        simulator.set_test_traffic_light_state(
                            "GREEN"
                        )

                        traffic_test_state = "GREEN"
                        traffic_test_state_started_at = (
                            traffic_now
                        )

                elif traffic_test_state == "GREEN":

                    green_semantics_ok = (
                        traffic_test_actor_verified
                        and
                        traffic_light_status["detected"]
                        and
                        traffic_light_status["state"]
                        == "GREEN"
                        and
                        traffic_light_status["risk"] == "LOW"
                        and
                        traffic_light_status["action"]
                        == "CONTINUE"
                        and
                        final_decision["final_risk"] == "LOW"
                        and
                        final_decision["final_action"]
                        == "CONTINUE"
                    )

                    green_agent_ok = (
                        agent_decision["agent_action"]
                        in {"CONTINUE", "WAIT"}
                    )

                    if (
                        green_semantics_ok
                        and green_agent_ok
                        and state_age
                        >= TRAFFIC_STATE_HOLD_SECONDS
                    ):

                        green_test_passed = True
                        traffic_test_state = (
                            "TRAFFIC_LIGHT_TEST_PASSED"
                        )

                        print("\n" + "=" * 70)
                        print("GREEN_TEST_PASSED")
                        print("GREEN -> LOW / CONTINUE")
                        print("=" * 70)
                        print(
                            "TRAFFIC_LIGHT_TEST_PASSED"
                        )
                        print(
                            "RED:    PASS"
                        )
                        print(
                            "YELLOW: PASS"
                        )
                        print(
                            "GREEN:  PASS"
                        )
                        print("=" * 70)

                        if not traffic_result_written["GREEN"]:
                            write_result_json(
                                "traffic_light_green_result.json",
                                compact_traffic_light_result(
                                    "GREEN", True, traffic_light_status,
                                    final_decision, control, agent_decision,
                                    traffic_test_actor_verified,
                                ),
                            )
                            traffic_result_written["GREEN"] = True

            # =================================================
            # 15. CARLA GROUND TRUTH
            # =================================================

            if simulator.npc_vehicle is not None:

                ground_truth = (
                    get_carla_ground_truth(
                        simulator.ego_vehicle,
                        simulator.npc_vehicle
                    )
                )

                true_npc_distance = (
                    ground_truth[
                        "true_distance_m"
                    ]
                )

            else:

                velocity = (
                    simulator.ego_vehicle.get_velocity()
                )

                speed_mps = (
                    velocity.x ** 2
                    + velocity.y ** 2
                    + velocity.z ** 2
                ) ** 0.5

                ground_truth = {
                    "speed_kmh": speed_mps * 3.6,
                    "true_distance_m": None
                }

                true_npc_distance = None

            ego_speed = (
                ground_truth[
                    "speed_kmh"
                ]
            )

            # =================================================
            # 16. VISION VEHICLE TELEMETRY
            # =================================================

            nearest_vision_vehicle = (
                find_nearest_vision_vehicle(
                    assessed_objects
                )
            )

            # =================================================
            # 17. RECOVERY STATE MACHINE
            # =================================================

            now = time.monotonic()

            # -------------------------------------------------

            if TEST_MODE == "NPC_RECOVERY":
                # A. VERIFY FULL STOP
                # -------------------------------------------------

                full_stop_verified = (
                    ego_speed < 0.1
                    and
                    agent_decision["state"]
                    in {
                        "STOPPED",
                        "REASSESSING"
                    }
                    and
                    final_decision[
                        "final_action"
                    ] == "STOP"
                )

                if (
                    recovery_state
                    == "APPROACHING"
                    and
                    full_stop_verified
                ):

                    recovery_state = (
                        "STOP_VERIFIED"
                    )

                    stop_started_at = now

                    print(
                        "\n"
                        + "=" * 70
                    )

                    print(
                        "RECOVERY TEST: "
                        "STOP VERIFIED"
                    )

                    print(
                        f"Ego Speed: "
                        f"{ego_speed:.2f} km/h"
                    )

                    print(
                        f"True NPC Distance: "
                        f"{true_npc_distance:.2f} m"
                    )

                    print(
                        "=" * 70
                    )

                # -------------------------------------------------
                # B. START CLEAR TIMER
                # -------------------------------------------------

                if (
                    recovery_state
                    == "STOP_VERIFIED"
                ):

                    recovery_state = (
                        "WAITING_TO_CLEAR"
                    )

                    print(
                        "\nWaiting "
                        f"{NPC_CLEAR_DELAY:.0f} "
                        "seconds before "
                        "clearing NPC..."
                    )

                # -------------------------------------------------
                # C. CLEAR NPC
                # -------------------------------------------------

                if (
                    recovery_state
                    == "WAITING_TO_CLEAR"
                    and
                    stop_started_at
                    is not None
                ):

                    stopped_for = (
                        now
                        - stop_started_at
                    )

                    if (
                        stopped_for
                        >= NPC_CLEAR_DELAY
                    ):

                        # Deterministically remove the controlled obstacle
                        # from the CARLA world instead of merely moving it
                        # farther ahead. Moving it to ~40 m can leave a
                        # MEDIUM/SLOW_DOWN vision target in the ego path,
                        # which correctly prevents the safety latch from
                        # accumulating safe confirmation frames.
                        #
                        # IMPORTANT: we do NOT reset or modify the safety
                        # supervisor. It must release naturally after its
                        # normal hold period + safe-frame confirmation.
                        npc_actor = simulator.npc_vehicle

                        if npc_actor is not None:
                            try:
                                npc_actor.destroy()
                                simulator.npc_vehicle = None
                                npc_cleared = True
                            except RuntimeError as exc:
                                npc_cleared = False
                                print(
                                    "WARNING: controlled NPC could not "
                                    f"be destroyed: {exc}"
                                )
                        else:
                            # Already absent means the controlled hazard
                            # is physically cleared from the test scene.
                            npc_cleared = True

                        if npc_cleared:

                            recovery_state = (
                                "WAITING_FOR_RESUME"
                            )

                            resume_started_at = None

                            # Discard stale evidence from this transition
                            # frame only. Normal perception resumes on the
                            # following frames, so any real remaining hazard
                            # can still block recovery.
                            assessed_objects = []

                            print(
                                "\n"
                                + "=" * 70
                            )

                            print(
                                "CONTROLLED NPC REMOVED"
                            )

                            print(
                                "Safety supervisor remains unchanged."
                            )

                            print(
                                "Waiting for normal safe-frame release "
                                "and autonomous recovery..."
                            )

                            print(
                                "=" * 70
                            )

                # =================================================
                # VALID RECOVERY CONDITIONS
                # =================================================

                agent_is_driving = (
                    agent_decision[
                        "state"
                    ] == "DRIVING"
                )

                throttle_applied = (
                    control.throttle > 0.0
                )

                brake_released = (
                    control.brake < 0.1
                )

                ego_is_moving = (
                    ego_speed
                    >
                    MIN_RECOVERY_SPEED_KMH
                )

                not_high_risk = (
                    final_decision[
                        "final_risk"
                    ] != "HIGH"
                )

                not_stop_command = (
                    final_decision[
                        "final_action"
                    ] != "STOP"
                )

                valid_recovery_driving = (
                    agent_is_driving
                    and
                    throttle_applied
                    and
                    brake_released
                    and
                    ego_is_moving
                    and
                    not_high_risk
                    and
                    not_stop_command
                )

                # -------------------------------------------------
                # D. WAIT FOR RESUME
                # -------------------------------------------------

                if (
                    recovery_state
                    == "WAITING_FOR_RESUME"
                ):

                    if valid_recovery_driving:

                        recovery_state = (
                            "RESUME_CANDIDATE"
                        )

                        resume_started_at = now

                        print(
                            "\n"
                            + "=" * 70
                        )

                        print(
                            "RESUME CANDIDATE DETECTED"
                        )

                        print(
                            f"Ego Speed: "
                            f"{ego_speed:.2f} km/h"
                        )

                        print(
                            f"Throttle: "
                            f"{control.throttle:.2f}"
                        )

                        print(
                            "Starting sustained "
                            "recovery timer..."
                        )

                        print(
                            "=" * 70
                        )

                # -------------------------------------------------
                # E. SUSTAINED RECOVERY
                # -------------------------------------------------

                elif (
                    recovery_state
                    == "RESUME_CANDIDATE"
                ):

                    if valid_recovery_driving:

                        recovery_duration = (
                            now
                            - resume_started_at
                        )

                        if (
                            recovery_duration
                            >=
                            RECOVERY_DRIVE_SECONDS_REQUIRED
                        ):

                            recovery_state = (
                                "RECOVERY_PASSED"
                            )

                            recovery_passed_at = now

                            print(
                                "\n"
                                + "=" * 70
                            )

                            print(
                                "RECOVERY TEST PASSED"
                            )

                            print(
                                "Autonomous recovery "
                                "was sustained."
                            )

                            print(
                                f"Driving Duration: "
                                f"{recovery_duration:.2f} s"
                            )

                            print(
                                f"Ego Speed: "
                                f"{ego_speed:.2f} km/h"
                            )

                            print(
                                f"Throttle: "
                                f"{control.throttle:.2f}"
                            )

                            print(
                                f"Brake: "
                                f"{control.brake:.2f}"
                            )

                            print(
                                "=" * 70
                            )

                            if not recovery_result_written:
                                write_result_json(
                                    "autonomous_recovery_result.json",
                                    {
                                        "status": "PASS",
                                        "test_mode": TEST_MODE,
                                        "stop_verified": True,
                                        "npc_cleared": bool(npc_cleared),
                                        "npc_clear_method": "DESTROY_ACTOR",
                                        "safety_supervisor_reset": False,
                                        "required_sustained_driving_seconds": (
                                            RECOVERY_DRIVE_SECONDS_REQUIRED
                                        ),
                                        "sustained_driving_seconds": float(
                                            recovery_duration
                                        ),
                                        "minimum_recovery_speed_kmh": (
                                            MIN_RECOVERY_SPEED_KMH
                                        ),
                                        "ego_speed_kmh_at_pass": float(ego_speed),
                                        "throttle_at_pass": float(control.throttle),
                                        "brake_at_pass": float(control.brake),
                                        "agent_state_at_pass": agent_decision.get("state"),
                                        "agent_action_at_pass": agent_decision.get(
                                            "agent_action"
                                        ),
                                        "final_risk_at_pass": final_decision.get(
                                            "final_risk"
                                        ),
                                        "final_action_at_pass": final_decision.get(
                                            "final_action"
                                        ),
                                        "validation_scope": (
                                            "Controlled obstacle stop, obstacle "
                                            "clearance, and sustained autonomous recovery"
                                        ),
                                    },
                                )
                                recovery_result_written = True

                    else:

                        print(
                            "\n"
                            + "-" * 70
                        )

                        print(
                            "RESUME CANDIDATE RESET"
                        )

                        print(
                            "Continuous autonomous "
                            "driving was interrupted."
                        )

                        print(
                            f"Agent State: "
                            f"{agent_decision['state']}"
                        )

                        print(
                            f"Decision: "
                            f"{final_decision['final_risk']} / "
                            f"{final_decision['final_action']}"
                        )

                        print(
                            f"Source: "
                            f"{final_decision['decision_source']}"
                        )

                        print(
                            "-" * 70
                        )

                        recovery_state = (
                            "WAITING_FOR_RESUME"
                        )

                        resume_started_at = None

                # -------------------------------------------------
                # F. NEW HAZARD AFTER SUCCESS
                # -------------------------------------------------

                elif (
                    recovery_state
                    == "RECOVERY_PASSED"
                ):

                    hazard_after_recovery = (
                        final_decision[
                            "final_risk"
                        ] == "HIGH"
                        or
                        final_decision[
                            "final_action"
                        ] == "STOP"
                        or
                        agent_decision[
                            "agent_action"
                        ] in {
                            "STOP",
                            "WAIT"
                        }
                    )

                    if (
                        hazard_after_recovery
                        and
                        not
                        new_hazard_after_recovery
                    ):

                        new_hazard_after_recovery = True

                        new_hazard_reason = (
                            f"{final_decision['final_risk']} / "
                            f"{final_decision['final_action']} / "
                            f"{agent_decision['agent_action']}"
                        )

                        print(
                            "\n"
                            + "=" * 70
                        )

                        print(
                            "NEW HAZARD AFTER RECOVERY"
                        )

                        print(
                            "Recovery remains PASSED."
                        )

                        print(
                            f"Decision: "
                            f"{new_hazard_reason}"
                        )

                        print(
                            f"Source: "
                            f"{final_decision['decision_source']}"
                        )

                        print(
                            f"Ego Speed: "
                            f"{ego_speed:.2f} km/h"
                        )

                        print(
                            f"Brake: "
                            f"{control.brake:.2f}"
                        )

                        print(
                            "=" * 70
                        )

            # =================================================
            # RECOVERY TIMER
            # =================================================

            recovery_elapsed = 0.0

            if (
                recovery_state
                == "RESUME_CANDIDATE"
                and
                resume_started_at
                is not None
            ):

                recovery_elapsed = (
                    now
                    - resume_started_at
                )

            elif (
                TEST_MODE == "NPC_RECOVERY"
                and
                recovery_state
                == "RECOVERY_PASSED"
            ):

                recovery_elapsed = (
                    RECOVERY_DRIVE_SECONDS_REQUIRED
                )

            # =================================================
            # TERMINAL OUTPUT
            # =================================================

            print(
                "\n\nSMARTDRIVE CARLA AGENT"
            )

            print(
                "-" * 70
            )

            # -------------------------------------------------
            # TEST CONFIGURATION
            # -------------------------------------------------

            print(
                "TEST CONFIGURATION"
            )

            print(
                f"Mode: {TEST_MODE}"
            )

            print(
                "Traffic-Light Control: "
                + (
                    "ENABLED"
                    if final_decision.get(
                        "traffic_light_control_enabled",
                        True
                    )
                    else
                    "DISABLED FOR TEST"
                )
            )

            print(
                "-" * 70
            )

            # -------------------------------------------------
            # PHYSICAL OBJECTS
            # -------------------------------------------------

            if assessed_objects:

                print(
                    "PHYSICAL OBJECTS"
                )

                print(
                    "-" * 70
                )

                for obj in assessed_objects:

                    print(
                        f"ID:       "
                        f"{obj.get('track_id')}"
                    )

                    print(
                        f"Class:    "
                        f"{obj.get('class_name')}"
                    )

                    print(
                        f"Region:   "
                        f"{obj.get('region')}"
                    )

                    print(
                        f"Motion:   "
                        f"{obj.get('motion')}"
                    )

                    print(
                        f"Distance: "
                        f"{obj.get('distance_m')} m"
                    )

                    print(
                        f"Risk:     "
                        f"{obj.get('risk_level')}"
                    )

                    print(
                        f"Action:   "
                        f"{obj.get('recommended_action')}"
                    )

                    print(
                        "-" * 70
                    )

            else:

                print(
                    "PHYSICAL OBJECTS: NONE"
                )

                print(
                    "-" * 70
                )

            # -------------------------------------------------
            # VISION TRAFFIC LIGHTS
            # -------------------------------------------------

            print(
                "VISION TRAFFIC LIGHTS"
            )

            print(
                f"Detected Boxes: "
                f"{len(traffic_light_detections)}"
            )

            print(
                "Decision Authority: "
                "CARLA SEMANTIC STATE"
            )

            print(
                "-" * 70
            )

            # -------------------------------------------------
            # OBJECT DECISION
            # -------------------------------------------------

            print(
                "OBJECT DECISION"
            )

            print(
                f"Risk:   {object_risk}"
            )

            print(
                f"Action: {object_action}"
            )

            print(
                "-" * 70
            )

            # -------------------------------------------------
            # TRAFFIC LIGHT
            # -------------------------------------------------

            print(
                "TRAFFIC LIGHT"
            )

            print(
                f"Detected: "
                f"{traffic_light_status['detected']}"
            )

            print(
                f"State:    "
                f"{traffic_light_status['state']}"
            )

            if (
                traffic_light_status[
                    "distance_m"
                ] is not None
            ):

                print(
                    f"Distance: "
                    f"{traffic_light_status['distance_m']:.2f} m"
                )

            print(
                f"Risk:     "
                f"{traffic_light_status['risk']}"
            )

            print(
                f"Action:   "
                f"{traffic_light_status['action']}"
            )

            print(
                "-" * 70
            )

            # -------------------------------------------------
            # FINAL DECISION
            # -------------------------------------------------

            print(
                "FINAL DECISION"
            )

            print(
                f"Object:  "
                f"{object_risk} / "
                f"{object_action}"
            )

            print(
                f"Traffic: "
                f"{traffic_risk} / "
                f"{traffic_action}"
            )

            print(
                f"FINAL:   "
                f"{final_decision['final_risk']} / "
                f"{final_decision['final_action']}"
            )

            print(
                f"Source:  "
                f"{final_decision['decision_source']}"
            )

            print(
                "-" * 70
            )

            # -------------------------------------------------
            # PERCEPTION SAFETY
            # -------------------------------------------------

            print(
                "PERCEPTION SAFETY"
            )

            print(
                f"Latched:  "
                f"{final_decision.get('safety_latched', False)}"
            )

            print(
                f"Override: "
                f"{final_decision.get('safety_override', False)}"
            )

            print(
                f"Reason:   "
                f"{final_decision.get('safety_reason', 'NONE')}"
            )

            print(
                f"Safe Confirmation: "
                f"{final_decision.get('safe_frames', 0)}/"
                f"{final_decision.get('safe_frames_required', 3)}"
            )

            if final_decision.get(
                "last_hazard_reason"
            ):

                print(
                    f"Last Hazard: "
                    f"{final_decision['last_hazard_reason']}"
                )

            if (
                final_decision.get(
                    "hazard_age_seconds"
                )
                is not None
            ):

                print(
                    f"Hazard Age: "
                    f"{final_decision['hazard_age_seconds']:.2f} s"
                )

            print(
                "-" * 70
            )

            # -------------------------------------------------
            # DRIVING AGENT
            # -------------------------------------------------

            print(
                "DRIVING AGENT"
            )

            print(
                f"State:  "
                f"{agent_decision['previous_state']} "
                f"-> "
                f"{agent_decision['state']}"
            )

            print(
                f"Action: "
                f"{agent_decision['agent_action']}"
            )

            print(
                f"Reason: "
                f"{agent_decision['reason']}"
            )

            print(
                "-" * 70
            )

            # -------------------------------------------------
            # VEHICLE CONTROL
            # -------------------------------------------------

            print(
                "VEHICLE CONTROL"
            )

            print(
                f"Throttle: "
                f"{control.throttle:.2f}"
            )

            print(
                f"Brake:    "
                f"{control.brake:.2f}"
            )

            print(
                f"Steer:    "
                f"{control.steer:.2f}"
            )

            print(
                "-" * 70
            )

            # -------------------------------------------------
            # GROUND TRUTH
            # -------------------------------------------------

            print(
                "CARLA GROUND TRUTH"
            )

            print(
                f"Ego Speed: "
                f"{ego_speed:.2f} km/h"
            )

            if true_npc_distance is not None:

                print(
                    f"True Controlled NPC Distance: "
                    f"{true_npc_distance:.2f} m"
                )

            else:

                print(
                    "True Controlled NPC Distance: "
                    "N/A (TRAFFIC-LIGHT TEST)"
                )

            if (
                nearest_vision_vehicle
                is not None
            ):

                print(
                    f"Nearest Vision Track: "
                    f"{nearest_vision_vehicle.get('track_id')}"
                )

                print(
                    f"Nearest Vision Class: "
                    f"{nearest_vision_vehicle.get('class_name')}"
                )

                print(
                    f"Nearest Vision Region: "
                    f"{nearest_vision_vehicle.get('region')}"
                )

                print(
                    f"Nearest Vision Distance: "
                    f"{nearest_vision_vehicle.get('distance_m'):.2f} m"
                )

                print(
                    "Vision Association: "
                    "UNVERIFIED"
                )

            else:

                print(
                    "Nearest Vision Vehicle: NONE"
                )

            print(
                "-" * 70
            )

            # -------------------------------------------------
            # RECOVERY VALIDATION
            # -------------------------------------------------

            if TEST_MODE == "NPC_RECOVERY":

                print(
                    "RECOVERY VALIDATION"
                )

                print(
                    f"Test State: "
                    f"{recovery_state}"
                )

                print(
                    f"Sustained Driving: "
                    f"{recovery_elapsed:.2f}/"
                    f"{RECOVERY_DRIVE_SECONDS_REQUIRED:.2f} s"
                )

                if new_hazard_after_recovery:

                    print(
                        "Post-Recovery Hazard: YES"
                    )

                    print(
                        f"Hazard Decision: "
                        f"{new_hazard_reason}"
                    )

                else:

                    print(
                        "Post-Recovery Hazard: NO"
                    )

                print(
                    "-" * 70
                )

            if TEST_MODE == "TRAFFIC_LIGHT_TEST":

                print(
                    "TRAFFIC-LIGHT VALIDATION"
                )

                print(
                    f"Test State: "
                    f"{traffic_test_state}"
                )

                print(
                    f"RED:    "
                    f"{'PASS' if red_test_passed else 'WAIT'}"
                )

                print(
                    f"YELLOW: "
                    f"{'PASS' if yellow_test_passed else 'WAIT'}"
                )

                print(
                    f"GREEN:  "
                    f"{'PASS' if green_test_passed else 'WAIT'}"
                )

                print(
                    "-" * 70
                )

            if TEST_MODE == "PERCEPTION_DROPOUT":

                print(
                    "PERCEPTION DROPOUT VALIDATION"
                )

                print(
                    f"Test State: {dropout_state}"
                )

                print(
                    "Latch Verified: "
                    f"{dropout_latch_verified}"
                )

                print(
                    "Override Verified: "
                    f"{dropout_override_verified}"
                )

                print(
                    "STOP Verified: "
                    f"{dropout_stop_verified}"
                )

                print(
                    "NPC Cleared:      "
                    f"{dropout_npc_cleared}"
                )

                print(
                    "Release Verified: "
                    f"{dropout_release_verified}"
                )

                print(
                    "-" * 70
                )

            # =================================================
            # VISUALIZATION
            # =================================================

            annotated_frame = (
                PerceptionVisualizer.draw(
                    frame,
                    results
                )
            )

            overlay_lines = [

                f"SOURCE: CARLA RGB-D | FRAME: {rgbd_frame_id}",

                (
                    f"TEST MODE: "
                    f"{TEST_MODE}"
                ),

                (
                    "TL CONTROL: "
                    + (
                        "ON"
                        if final_decision.get(
                            "traffic_light_control_enabled",
                            True
                        )
                        else
                        "OFF (TEST)"
                    )
                ),

                (
                    f"TRACKED OBJECTS: "
                    f"{len(objects)}"
                ),

                (
                    f"PHYSICAL OBJECTS: "
                    f"{len(assessed_objects)}"
                ),

                (
                    f"VISION TL BOXES: "
                    f"{len(traffic_light_detections)}"
                ),

                (
                    f"FINAL RISK: "
                    f"{final_decision['final_risk']}"
                ),

                (
                    f"FINAL ACTION: "
                    f"{final_decision['final_action']}"
                ),

                (
                    f"DECISION SOURCE: "
                    f"{final_decision['decision_source']}"
                ),

                (
                    f"SAFETY LATCH: "
                    f"{'ON' if final_decision.get('safety_latched', False) else 'OFF'}"
                ),

                (
                    f"SAFETY OVERRIDE: "
                    f"{'YES' if final_decision.get('safety_override', False) else 'NO'}"
                ),

                (
                    f"SAFETY REASON: "
                    f"{final_decision.get('safety_reason', 'NONE')}"
                ),

                (
                    f"SAFE CONFIRM: "
                    f"{final_decision.get('safe_frames', 0)}/"
                    f"{final_decision.get('safe_frames_required', 3)}"
                ),

                (
                    f"TL STATE: "
                    f"{traffic_light_status['state']}"
                ),

                (
                    f"TL ACTION: "
                    f"{traffic_light_status['action']}"
                ),

                (
                    f"AGENT STATE: "
                    f"{agent_decision['state']}"
                ),

                (
                    f"AGENT ACTION: "
                    f"{agent_decision['agent_action']}"
                ),

                (
                    f"CONTROL: "
                    f"T={control.throttle:.2f} "
                    f"B={control.brake:.2f} "
                    f"S={control.steer:.2f}"
                ),

                (
                    f"EGO SPEED: "
                    f"{ego_speed:.2f} KM/H"
                ),

                (
                    "TRUE NPC DIST: "
                    + (
                        f"{true_npc_distance:.2f} M"
                        if true_npc_distance is not None
                        else "N/A"
                    )
                )
            ]

            # -------------------------------------------------
            # TL DISTANCE
            # -------------------------------------------------

            if (
                traffic_light_status[
                    "distance_m"
                ] is not None
            ):

                overlay_lines.append(
                    (
                        f"TL DIST: "
                        f"{traffic_light_status['distance_m']:.2f} M"
                    )
                )

            # -------------------------------------------------
            # NEAREST VEHICLE
            # -------------------------------------------------

            if (
                nearest_vision_vehicle
                is not None
            ):

                overlay_lines.append(
                    (
                        f"NEAREST VEHICLE: "
                        f"{nearest_vision_vehicle.get('distance_m'):.2f} M"
                    )
                )

            # -------------------------------------------------
            # RECOVERY OVERLAY
            # -------------------------------------------------

            if TEST_MODE == "TRAFFIC_LIGHT_TEST":

                overlay_lines.append(
                    (
                        f"TL TEST: "
                        f"{traffic_test_state}"
                    )
                )

                overlay_lines.append(
                    (
                        "TL CHECKS: "
                        f"R={int(red_test_passed)} "
                        f"Y={int(yellow_test_passed)} "
                        f"G={int(green_test_passed)}"
                    )
                )

            if TEST_MODE == "NPC_RECOVERY":

                overlay_lines.append(
                    (
                        f"RECOVERY: "
                        f"{recovery_state}"
                    )
                )

            elif TEST_MODE == "PERCEPTION_DROPOUT":

                overlay_lines.append(
                    (
                        f"DROPOUT: "
                        f"{dropout_state}"
                    )
                )

                overlay_lines.append(
                    (
                        "DROPOUT CHECKS: "
                        f"L={int(dropout_latch_verified)} "
                        f"O={int(dropout_override_verified)} "
                        f"S={int(dropout_stop_verified)} "
                        f"C={int(dropout_npc_cleared)} "
                        f"R={int(dropout_release_verified)}"
                    )
                )

            if (
                TEST_MODE == "NPC_RECOVERY"
                and
                recovery_state
                == "RESUME_CANDIDATE"
            ):

                overlay_lines.append(
                    (
                        f"RECOVERY TIMER: "
                        f"{recovery_elapsed:.2f}/"
                        f"{RECOVERY_DRIVE_SECONDS_REQUIRED:.2f} S"
                    )
                )

            elif (
                recovery_state
                == "RECOVERY_PASSED"
            ):

                overlay_lines.append(
                    "RECOVERY TIMER: PASSED"
                )

            if (
                TEST_MODE == "NPC_RECOVERY"
                and new_hazard_after_recovery
            ):

                overlay_lines.append(
                    "POST-RECOVERY HAZARD: YES"
                )

            if (
                TEST_MODE == "ROUTE_NAVIGATION_TEST"
                and route_test_finished
            ):
                overlay_lines.append("ROUTE TEST: PASSED")
                overlay_lines.append("VEHICLE: STOPPED")
                overlay_lines.append("PRESS Q TO EXIT")

            # -------------------------------------------------
            # PERCEPTION SAFETY AGE
            # -------------------------------------------------

            if (
                final_decision.get(
                    "hazard_age_seconds"
                )
                is not None
            ):

                overlay_lines.append(
                    (
                        f"HAZARD AGE: "
                        f"{final_decision['hazard_age_seconds']:.2f} S"
                    )
                )

            # =================================================
            # DRAW TEXT
            # =================================================

            y = 30

            for line in overlay_lines:

                cv2.putText(
                    annotated_frame,
                    line,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                y += 28

            # =================================================
            # DISPLAY
            # =================================================

            cv2.imshow(
                "SmartDrive | CARLA Agent",
                annotated_frame
            )

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):

                break

    except KeyboardInterrupt:

        print(
            "\nStopping CARLA agent..."
        )

    finally:

        simulator.destroy()
        if trajectory_log is not None:
            trajectory_log.close()
            summary = evaluate_trajectory(trajectory_records, Path(__file__).parent / "results")
            print("TRAJECTORY_PREDICTION_EVALUATED" if summary["status"] == "EVALUATED" else "TRAJECTORY_PREDICTION_INSUFFICIENT_DATA")
            print(json.dumps(summary, indent=2))
        if TEST_MODE == "ROUTE_NAVIGATION_TEST":
            result = route_navigator.report() if route_navigator is not None else {"status": "NOT_PASSED", "route_generated": False}
            result.update(outcome=route_outcome, start_index=ROUTE_START_INDEX)
            result_dir = Path(__file__).parent / "results"
            result_dir.mkdir(exist_ok=True)
            (result_dir / "route_navigation_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if TEST_MODE in RGBD_VALIDATION_MODES:
            payload = {
                "status": "PASS" if distance_validation.passed else "NOT_PASSED",
                "test_mode": TEST_MODE,
                "reference": "SEMANTIC_LIDAR_CAMERA_SURFACE",
                "coordinate_convention": "X right, Y down, Z forward; meters",
                "axis_tolerance_m": (
                    AXIS_TOLERANCE_M
                    if TEST_MODE == "RGBD_3D_LOCALIZATION_TEST"
                    else None
                ),
                "position_tolerance_m": (
                    POSITION_TOLERANCE_M
                    if TEST_MODE == "RGBD_3D_LOCALIZATION_TEST"
                    else None
                ),
                "tolerance": (
                    "fixed per-axis and Euclidean"
                    if TEST_MODE == "RGBD_3D_LOCALIZATION_TEST"
                    else "max(0.5 m, 5% of reference)"
                ),
                "required_consecutive_frames": 10,
                "results": distance_validation.results,
            }

            benchmark_name = (
                "rgbd_3d_localization_result.json"
                if TEST_MODE == "RGBD_3D_LOCALIZATION_TEST"
                else "rgbd_distance_result.json"
            )
            report = write_result_json(benchmark_name, payload)
            print(f"RGB-D validation report: {report}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
