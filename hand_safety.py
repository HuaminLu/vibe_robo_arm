"""
hand_safety.py - The Lifeguard
Monitors camera feed for human hands in the danger zone.
Immediately triggers emergency retract if hand detected.
Runs continuously in background.
"""

import cv2
import numpy as np
import threading
import time
from collections import deque


class SafetyMonitor:
    """Monitors for human hands and enforces safety zones"""
    
    def __init__(self, robot_controller, danger_zone_radius=100, timeout=5):
        """
        Initialize safety monitor
        
        Args:
            robot_controller: Reference to RobotController for emergency stops
            danger_zone_radius: Radius of danger zone around robot workspace
            timeout: Seconds to wait before allowing operation to resume after hand detected
        """
        self.robot_controller = robot_controller
        self.danger_zone_radius = danger_zone_radius
        self.danger_zone_center = (150, 150)  # Center of robot workspace
        self.timeout = timeout
        
        self.running = True
        self.monitoring = True
        self.hand_detected = False
        self.last_hand_detection_time = 0
        
        # Hand detection configuration
        self.hand_detector_type = "skin_color"  # Options: "skin_color", "yolo"
        self.hand_history = deque(maxlen=10)  # Keep history for smoothing
        
    def run(self):
        """Main safety monitoring loop (runs in background thread)"""
        print("[SAFETY] Safety monitor started")
        
        while self.running:
            if not self.monitoring:
                time.sleep(0.1)
                continue
            
            # Check if we should allow operation to resume
            if self.hand_detected:
                elapsed = time.time() - self.last_hand_detection_time
                if elapsed > self.timeout:
                    print(f"[SAFETY] Timeout reached, resuming operations")
                    self.hand_detected = False
            
            time.sleep(0.05)
    
    def check_frame_for_hands(self, frame):
        """
        Check camera frame for hands in danger zone
        
        Args:
            frame: Camera frame (BGR)
            
        Returns:
            True if hand detected, False otherwise
        """
        if not self.monitoring:
            return False
        
        hands = self._detect_hands(frame)
        hand_in_danger_zone = self._check_danger_zone(hands, frame.shape)
        
        if hand_in_danger_zone:
            self._handle_hand_detected()
            return True
        
        return False
    
    def _detect_hands(self, frame):
        """
        Detect hands in frame using configured method
        
        Args:
            frame: Camera frame
            
        Returns:
            List of hand positions [(x, y, confidence), ...]
        """
        if self.hand_detector_type == "skin_color":
            return self._detect_hands_color(frame)
        elif self.hand_detector_type == "yolo":
            return self._detect_hands_yolo(frame)
        else:
            return []
    
    def _detect_hands_color(self, frame):
        """Detect hands using skin color detection"""
        # Convert to HSV for skin color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        # Adjust these ranges based on lighting conditions
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        hands = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Hand area typically between 1000 and 50000 pixels
            if area > 1000:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    x = int(M["m10"] / M["m00"])
                    y = int(M["m01"] / M["m00"])
                    confidence = min(1.0, area / 10000)
                    hands.append((x, y, confidence))
        
        return hands
    
    def _detect_hands_yolo(self, frame):
        """Detect hands using YOLO object detection"""
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")  # Load pretrained model
            results = model(frame, classes=0)  # 0 = person class
            
            hands = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x = (x1 + x2) / 2
                    y = (y1 + y2) / 2
                    conf = box.conf[0]
                    hands.append((float(x), float(y), float(conf)))
            
            return hands
        except ImportError:
            print("[SAFETY] YOLO not available, using color detection")
            return self._detect_hands_color(frame)
    
    def _check_danger_zone(self, hands, frame_shape):
        """
        Check if any detected hands are in the danger zone
        
        Args:
            hands: List of detected hands [(x, y, confidence), ...]
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            True if hand in danger zone, False otherwise
        """
        if not hands:
            return False
        
        height, width = frame_shape[:2]
        
        # Scale danger zone to frame size
        # Assuming frame is ~640x480 and danger zone is in center
        for x, y, confidence in hands:
            # Calculate distance from danger zone center
            dx = x - self.danger_zone_center[0]
            dy = y - self.danger_zone_center[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # If hand is within danger zone and confidence is high enough
            if distance < self.danger_zone_radius and confidence > 0.5:
                self.hand_history.append((x, y, confidence, True))
                return True
        
        self.hand_history.append((None, None, None, False))
        return False
    
    def _handle_hand_detected(self):
        """Handle detection of hand in danger zone"""
        current_time = time.time()
        self.hand_detected = True
        self.last_hand_detection_time = current_time
        
        print("[SAFETY] ⚠ HAND DETECTED IN DANGER ZONE!")
        print("[SAFETY] Initiating emergency stop")
        
        # Trigger emergency stop on robot
        self.robot_controller.emergency_stop()
    
    def set_danger_zone(self, center_x, center_y, radius):
        """
        Configure danger zone
        
        Args:
            center_x, center_y: Center of danger zone
            radius: Radius of danger zone
        """
        self.danger_zone_center = (center_x, center_y)
        self.danger_zone_radius = radius
        print(f"[SAFETY] Danger zone updated: center=({center_x}, {center_y}), radius={radius}")
    
    def visualize_danger_zone(self, frame):
        """
        Draw danger zone on frame for visualization
        
        Args:
            frame: Camera frame
            
        Returns:
            Frame with danger zone drawn
        """
        annotated = frame.copy()
        
        # Draw danger zone circle
        zone_color = (0, 0, 255) if self.hand_detected else (0, 255, 0)
        cv2.circle(annotated, self.danger_zone_center, self.danger_zone_radius, 
                  zone_color, 2)
        
        # Label
        status = "HAND DETECTED!" if self.hand_detected else "SAFE"
        cv2.putText(annotated, f"Safety: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, zone_color, 2)
        
        return annotated
    
    def enable_monitoring(self):
        """Enable safety monitoring"""
        self.monitoring = True
        print("[SAFETY] Monitoring enabled")
    
    def disable_monitoring(self):
        """Disable safety monitoring (not recommended)"""
        self.monitoring = False
        print("[SAFETY] Monitoring disabled")
    
    def get_status(self):
        """Get current safety status"""
        return {
            "monitoring": self.monitoring,
            "hand_detected": self.hand_detected,
            "danger_zone": self.danger_zone_center,
            "timeout_remaining": max(0, self.timeout - (time.time() - self.last_hand_detection_time)) if self.hand_detected else 0
        }
    
    def stop(self):
        """Stop the safety monitor"""
        self.running = False
        print("[SAFETY] Safety monitor stopped")


class SkinColorDetector:
    """Dedicated skin color detection class"""
    
    def __init__(self):
        """Initialize skin color detector"""
        # HSV ranges for skin color (can be adjusted)
        self.lower_hsv = np.array([0, 20, 70])
        self.upper_hsv = np.array([20, 255, 255])
    
    def get_skin_mask(self, frame):
        """Get binary mask of skin regions"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def detect_hands(self, frame):
        """Detect hand locations"""
        mask = self.get_skin_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        hands = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    x = int(M["m10"] / M["m00"])
                    y = int(M["m01"] / M["m00"])
                    hands.append((x, y))
        
        return hands
