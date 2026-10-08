import cv2

from multimodal.vlm import VisionLanguageModel
from multimodal.scene_understanding import SceneUnderstanding


# -------------------------------------------------
# INITIALIZE VLM
# -------------------------------------------------
vlm = VisionLanguageModel()
scene_understanding = SceneUnderstanding(vlm)


# -------------------------------------------------
# CAPTURE ONE FRAME
# -------------------------------------------------
camera = cv2.VideoCapture(0)

success, frame = camera.read()

camera.release()

if not success:
    raise RuntimeError("Could not capture camera frame.")


# -------------------------------------------------
# ANALYZE SCENE
# -------------------------------------------------
print("\nAnalyzing scene...\n")

scene = scene_understanding.analyze(frame)


# -------------------------------------------------
# PRINT STRUCTURED RESULT
# -------------------------------------------------
print("SMARTDRIVE MULTIMODAL SCENE")
print("-" * 70)

print(f"Scene:   {scene['scene']}")
print(f"Hazards: {scene['hazards']}")
print(f"Risk:    {scene['risk']}")
print(f"Action:  {scene['recommended_action']}")