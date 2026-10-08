import time

from autonomy.driving_agent import DrivingAgent


agent = DrivingAgent(
    stop_hold_seconds=2.0,
    safe_frames_required=3
)


def run_test(name, risk, action):

    decision = {
        "final_risk": risk,
        "final_action": action
    }

    result = agent.step(
        decision
    )

    print("\n" + name)
    print("-" * 70)

    print(
        f"Input: "
        f"{risk} / {action}"
    )

    print(
        f"State: "
        f"{result['previous_state']} "
        f"-> "
        f"{result['state']}"
    )

    print(
        f"Agent Action: "
        f"{result['agent_action']}"
    )

    print(
        f"Reason: "
        f"{result['reason']}"
    )


# -------------------------------------------------
# TEST 1
# Normal driving
# -------------------------------------------------
run_test(
    "TEST 1 - SAFE",
    "LOW",
    "CONTINUE"
)


# -------------------------------------------------
# TEST 2
# Hazard appears
# -------------------------------------------------
run_test(
    "TEST 2 - HAZARD",
    "HIGH",
    "STOP"
)


# -------------------------------------------------
# TEST 3
# Hazard disappears immediately
# Agent should still WAIT
# -------------------------------------------------
run_test(
    "TEST 3 - EARLY CLEAR",
    "LOW",
    "CONTINUE"
)


# Wait until minimum stop duration expires.
time.sleep(2.2)


# -------------------------------------------------
# TEST 4
# Begin reassessment
# -------------------------------------------------
run_test(
    "TEST 4 - REASSESS",
    "LOW",
    "CONTINUE"
)


# -------------------------------------------------
# TEST 5
# Second safe confirmation
# -------------------------------------------------
run_test(
    "TEST 5 - SAFE CONFIRMATION",
    "LOW",
    "CONTINUE"
)


# -------------------------------------------------
# TEST 6
# Third safe confirmation
# Agent should RESUME
# -------------------------------------------------
run_test(
    "TEST 6 - RESUME",
    "LOW",
    "CONTINUE"
)