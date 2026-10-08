from autonomy.decision_fusion import DecisionFusion


fusion = DecisionFusion()


# Simulate current SmartDrive object result
assessed_objects = [
    {
        "track_id": 2,
        "class_name": "person",
        "risk_level": "HIGH",
        "recommended_action": "STOP"
    }
]


# Simulate current VLM result
vlm_scene = {
    "scene": "A person is sitting on a couch.",
    "hazards": "None",
    "risk": "LOW",
    "recommended_action": "CONTINUE"
}


decision = fusion.fuse(
    assessed_objects,
    vlm_scene
)


print("\nMULTIMODAL DECISION FUSION")
print("-" * 80)

print(
    f"Object: "
    f"{decision['object_risk']} / "
    f"{decision['object_action']}"
)

print(
    f"VLM:    "
    f"{decision['vlm_risk']} / "
    f"{decision['vlm_action']}"
)

print(
    f"FINAL:  "
    f"{decision['final_risk']} / "
    f"{decision['final_action']}"
)

print(
    f"Reason: {decision['reason']}"
)