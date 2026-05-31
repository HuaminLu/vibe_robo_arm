"""
camera_overlay_selection.py - Click-to-Grab
Listens for mouse clicks on the live video feed.
When clicking inside a detection box, grabs that Lego's location and sends to robot.
"""

import cv2
import numpy as np
import os
import time
import tempfile
import urllib.request
from PyQt5.QtCore import Qt

try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import base_options
    from mediapipe.tasks.python.vision.core.image import Image
    MP_HANDS_TASKS_AVAILABLE = True
except Exception:
    MP_HANDS_TASKS_AVAILABLE = False

try:
    import mediapipe as mp
    MP_HANDS_SOLUTIONS_AVAILABLE = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
except ImportError:
    MP_HANDS_SOLUTIONS_AVAILABLE = False

class ClickHandler:
    """Handles mouse click events on video feed for object selection"""
    
    def __init__(self, on_selection_callback=None):
        """
        Initialize click handler
        
        Args:
            on_selection_callback: Function to call when object selected
                                  Signature: callback(lego_id, location)
        """
        self.on_selection_callback = on_selection_callback
        self.current_detections = []
        self.last_frame = None
        self.selected_history = []
        self.click_markers = []
        self.mode = "Manual"
    
    def set_detections(self, detections, frame):
        """
        Update current detections and frame
        
        Args:
            detections: List of detection dictionaries
            frame: Current camera frame
        """
        self.current_detections = detections
        self.last_frame = frame
    
    def handle_click(self, click_x, click_y, display_width, display_height):
        """
        Handle mouse click on video display
        
        Args:
            click_x, click_y: Click coordinates in display space
            display_width: Width of display widget
            display_height: Height of display widget
        """
        if self.last_frame is None:
            print("[CLICK] No frame available")
            return

        if self.mode == "Automatic":
            print("[CLICK] Ignoring click in Automatic mode")
            return

        # Convert display coordinates to frame coordinates
        frame_height, frame_width = self.last_frame.shape[:2]
        scale_x = frame_width / display_width
        scale_y = frame_height / display_height
        
        frame_x = int(click_x * scale_x)
        frame_y = int(click_y * scale_y)
        
        # Find which detection was clicked
        selected_detection = self._find_clicked_detection(frame_x, frame_y)
        
        if selected_detection:
            self._handle_selection(selected_detection)
            self._add_click_marker((selected_detection["x"], selected_detection["y"]))
        else:
            print(f"[CLICK] No object at ({frame_x}, {frame_y})")
            self._add_click_marker((frame_x, frame_y))
    
    def _find_clicked_detection(self, click_x, click_y):
        """
        Find which detection was clicked
        
        Args:
            click_x, click_y: Click coordinates in frame space
            
        Returns:
            Detection dictionary or None
        """
        for detection in self.current_detections:
            x = detection["x"]
            y = detection["y"]
            width = detection["width"]
            height = detection["height"]
            
            # Check if click is within bounding box
            left = x - width / 2
            right = x + width / 2
            top = y - height / 2
            bottom = y + height / 2
            
            if left <= click_x <= right and top <= click_y <= bottom:
                return detection
        
        return None
    
    def _handle_selection(self, detection):
        """
        Handle selection of a detection
        
        Args:
            detection: Selected detection dictionary
        """
        lego_id = detection["id"]
        location = (detection["x"], detection["y"])
        confidence = detection["confidence"]
        
        print(f"[CLICK] Selected Lego {lego_id} at {location} (confidence: {confidence:.2f})")
        
        # Record in history
        self.selected_history.append({
            "lego_id": lego_id,
            "location": location,
            "confidence": confidence
        })
        
        # Call callback if provided
        if self.on_selection_callback:
            self.on_selection_callback(lego_id, location)
    
    def get_selection_history(self):
        """Get history of selections"""
        return self.selected_history.copy()
    
    def set_mode(self, mode):
        """Set current click mode and adjust marker behaviour"""
        self.mode = mode
        if mode == "Automatic":
            self.clear_click_markers()
    
    def clear_click_markers(self):
        """Remove any active click markers"""
        self.click_markers.clear()
    
    def get_active_click_markers(self):
        """Return active click markers and drop expired markers"""
        if self.mode == "Custom":
            return self.click_markers.copy()

        now = time.time()
        self.click_markers = [m for m in self.click_markers if now - m["created"] < m["duration"]]
        return self.click_markers.copy()
    
    def clear_history(self):
        """Clear selection history"""
        self.selected_history.clear()
        self.click_markers.clear()
    
    def _add_click_marker(self, position):
        """Add a temporary click marker for feedback"""
        if self.mode == "Automatic":
            return

        if self.mode == "Manual":
            self.click_markers.clear()
            duration = 2.0
        elif self.mode == "Custom":
            if len(self.click_markers) >= 4:
                self.click_markers.pop(0)
            duration = None
        else:
            duration = 2.0

        marker = {
            "position": (int(position[0]), int(position[1])),
            "created": time.time(),
            "duration": duration
        }
        self.click_markers.append(marker)


class HandGestureDetector:
    """Detects hand landmarks and gestures using MediaPipe"""

    MODEL_FILE_NAME = "mediapipe_hand_landmarker.task"
    MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    def __init__(self, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5):
        self.enabled = False
        self.latest_hands = []
        self.mode = None
        self.landmarker = None
        self.hands = None

        if MP_HANDS_TASKS_AVAILABLE:
            self.mode = "tasks"
            try:
                self._init_tasks_detector(max_num_hands, min_detection_confidence, min_tracking_confidence)
                self.enabled = True
            except Exception as e:
                print(f"[HAND] MediaPipe Tasks initialization failed: {e}")
                self.enabled = False

        elif MP_HANDS_SOLUTIONS_AVAILABLE:
            self.mode = "solutions"
            self._init_solutions_detector(max_num_hands, min_detection_confidence, min_tracking_confidence)
            self.enabled = True

    def _init_tasks_detector(self, max_num_hands, min_detection_confidence, min_tracking_confidence):
        model_path = self._ensure_model_available()
        options = vision.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=model_path),
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def _ensure_model_available(self):
        model_path = os.path.join(tempfile.gettempdir(), self.MODEL_FILE_NAME)
        if not os.path.exists(model_path):
            print(f"[HAND] Downloading MediaPipe hand model to {model_path}")
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
        return model_path

    def _init_solutions_detector(self, max_num_hands, min_detection_confidence, min_tracking_confidence):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame):
        self.latest_hands = []
        if not self.enabled:
            return []

        if self.mode == "tasks":
            return self._process_tasks_frame(frame)

        if self.mode == "solutions":
            return self._process_solutions_frame(frame)

        return []

    def _process_tasks_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_file.close()
        try:
            cv2.imwrite(temp_file.name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            image = Image.create_from_file(temp_file.name)
            result = self.landmarker.detect(image)
        except Exception as e:
            print(f"[HAND] Detection failed: {e}")
            return []
        finally:
            try:
                os.remove(temp_file.name)
            except OSError:
                pass

        return self._parse_task_result(result, frame)

    def _parse_task_result(self, result, frame):
        if not result or not result.hand_landmarks:
            return []

        height, width = frame.shape[:2]
        hands = []
        for landmarks, handedness_list in zip(result.hand_landmarks, result.handedness):
            landmark_points = []
            for idx, lm in enumerate(landmarks):
                landmark_points.append({
                    "id": idx,
                    "x": int(lm.x * width),
                    "y": int(lm.y * height),
                    "z": lm.z,
                })

            label = "Hand"
            if handedness_list:
                category = handedness_list[0]
                label = getattr(category, "category_name", None) or getattr(category, "label", None) or label

            gesture = self._classify_gesture(landmark_points, label)
            hands.append({
                "landmarks": landmark_points,
                "gesture": gesture,
                "handedness": label,
            })

        return hands

    def _process_solutions_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return []

        height, width = frame.shape[:2]
        hands = []
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmarks = []
            for idx, lm in enumerate(hand_landmarks.landmark):
                landmarks.append({
                    "id": idx,
                    "x": int(lm.x * width),
                    "y": int(lm.y * height),
                    "z": lm.z,
                })

            handedness_label = handedness.classification[0].label
            gesture = self._classify_gesture(landmarks, handedness_label)
            hands.append({
                "landmarks": landmarks,
                "gesture": gesture,
                "handedness": handedness_label,
            })

        return hands

    def _finger_is_extended(self, landmarks, tip_id, pip_id, handedness):
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        if tip_id == 4:
            if handedness == "Right":
                return tip["x"] > pip["x"]
            return tip["x"] < pip["x"]
        return tip["y"] < pip["y"]

    def _classify_gesture(self, landmarks, handedness):
        extended = [
            self._finger_is_extended(landmarks, 4, 3, handedness),
            self._finger_is_extended(landmarks, 8, 6, handedness),
            self._finger_is_extended(landmarks, 12, 10, handedness),
            self._finger_is_extended(landmarks, 16, 14, handedness),
            self._finger_is_extended(landmarks, 20, 18, handedness),
        ]
        count = sum(extended)
        if count == 0:
            return "Fist"
        if count == 5:
            return "Open"
        if count == 1 and extended[1]:
            return "Point"
        if count == 2 and extended[1] and extended[2]:
            return "Peace"
        return "Gesture"


class OverlayRenderer:
    """Renders interactive overlay on camera feed"""
    
    def __init__(self):
        """Initialize overlay renderer"""
        self.highlight_color = (0, 255, 0)  # Green for normal
        self.hover_color = (255, 255, 0)    # Yellow for hover
        self.selected_color = (0, 0, 255)   # Red for selected
        self.line_thickness = 2
    
    def draw_interactive_boxes(self, frame, detections, hover_detection=None, selected_detection=None, click_markers=None, mode=None):
        """
        Draw interactive bounding boxes on frame
        
        Args:
            frame: Camera frame
            detections: List of detections
            hover_detection: Detection under mouse hover
            selected_detection: Currently selected detection
            click_markers: List of temporary click markers
            mode: Current overlay mode
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        if click_markers:
            if mode == "Custom" and len(click_markers) > 1:
                for start, end in zip(click_markers, click_markers[1:]):
                    start_pos = start["position"]
                    end_pos = end["position"]
                    cv2.line(annotated, start_pos, end_pos, (255, 0, 255), 2)

            for idx, marker in enumerate(click_markers, start=1):
                px, py = marker["position"]
                cv2.circle(annotated, (px, py), 10, (255, 0, 0), -1)
                cv2.circle(annotated, (px, py), 14, (255, 0, 0), 2)
                if mode == "Custom":
                    cv2.putText(
                        annotated,
                        str(idx),
                        (px - 8, py + 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

        return annotated

    def draw_hand_skeletons(self, frame, hand_data):
        """Draw hand skeleton nodes and gesture labels"""
        annotated = frame.copy()
        if not hand_data:
            return annotated

        for hand in hand_data:
            landmarks = hand.get("landmarks", [])
            for start_idx, end_idx in HandGestureDetector.CONNECTIONS:
                if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                    continue
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                cv2.line(annotated, (start["x"], start["y"]), (end["x"], end["y"]), (0, 255, 255), 2)

            for lm in landmarks:
                cv2.circle(annotated, (lm["x"], lm["y"]), 4, (0, 255, 255), -1)

            if landmarks:
                wrist = landmarks[0]
                label = f"{hand.get('handedness', 'Hand')} {hand.get('gesture', '')}"
                cv2.putText(annotated, label, (wrist["x"] + 5, wrist["y"] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return annotated
    
    def draw_selection_feedback(self, frame, selected_detection):
        """
        Draw feedback for selected object
        
        Args:
            frame: Camera frame
            selected_detection: Selected detection
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        if selected_detection:
            x = int(selected_detection["x"])
            y = int(selected_detection["y"])
            w = int(selected_detection["width"] / 2)
            h = int(selected_detection["height"] / 2)
            
            # Draw pulsing box (animation effect)
            cv2.rectangle(annotated, (x - w, y - h), (x + w, y + h), (0, 0, 255), 3)
            cv2.rectangle(annotated, (x - w - 5, y - h - 5), (x + w + 5, y + h + 5), (0, 0, 255), 1)
            
            # Draw status text
            status = "TARGET LOCKED - Ready to pick"
            cv2.putText(annotated, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return annotated


class CameraOverlay:
    """Main overlay system combining click detection and rendering"""
    
    def __init__(self, on_selection_callback=None):
        """
        Initialize camera overlay
        
        Args:
            on_selection_callback: Callback for object selection
        """
        self.click_handler = ClickHandler(on_selection_callback)
        self.overlay_renderer = OverlayRenderer()
        self.hand_detector = HandGestureDetector()
        self.hand_data = []
        self.selected_detection = None
        self.hover_detection = None
        self.mode = "Manual"
    
    def update_detections(self, detections, frame):
        """Update current detections and frame"""
        self.click_handler.set_detections(detections, frame)
        self.hand_data = self.hand_detector.process_frame(frame)
    
    def handle_click(self, click_x, click_y, display_width, display_height):
        """Handle mouse click"""
        self.click_handler.handle_click(click_x, click_y, display_width, display_height)
        
        # Update selected detection
        if self.click_handler.selected_history:
            last_selection = self.click_handler.selected_history[-1]
            self.selected_detection = {
                "id": last_selection["lego_id"],
                "x": last_selection["location"][0],
                "y": last_selection["location"][1]
            }
    
    def render_frame(self, frame, detections):
        """
        Render interactive overlay on frame
        
        Args:
            frame: Camera frame
            detections: List of detections
            
        Returns:
            Annotated frame
        """
        hand_frame = self.overlay_renderer.draw_hand_skeletons(frame, self.hand_data)
        click_markers = self.click_handler.get_active_click_markers()
        return self.overlay_renderer.draw_interactive_boxes(
            hand_frame, detections, self.hover_detection, self.selected_detection, click_markers
        )

    def set_mode(self, mode):
        """Set the interaction mode for overlay click handling"""
        self.mode = mode
        self.click_handler.set_mode(mode)
    
    def get_selected_object(self):
        """Get currently selected object"""
        return self.selected_detection
    
    def clear_selection(self):
        """Clear current selection"""
        self.selected_detection = None
