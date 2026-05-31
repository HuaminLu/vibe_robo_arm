import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Configure the Hand Landmarker Task
model_path = 'hand_landmarker.task'  # Make sure this file is in your folder!

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE # Optimizes processing frame-by-frame
)

# Create the detector instance
detector = vision.HandLandmarker.create_from_options(options)

# Initialize Camera
cap = cv2.VideoCapture(1)

print("Starting safety feed. Press 'q' to exit...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipe Tasks require converted mp.Image format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 2. Run Hand Detection
    detection_result = detector.detect(mp_image)

    # 3. Process Safety Logic
    # If the hand_landmarks list inside the result is not empty, a hand is present!
    if detection_result.hand_landmarks:
        print("[ALERT] Human Hand Detected! Safety Halt Active.")
        cv2.putText(frame, "!! EMERGENCY STOP: HAND IN WORKSPACE !!", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Status: Safe (No Hands)", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the live window
    cv2.imshow("Safety Monitor Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()