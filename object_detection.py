"""
object_detection.py - Lego Finder
Uses YOLO AI model to detect Lego pieces in camera feed.
Identifies location and size of each detected piece on the table.
"""

import cv2
import numpy as np
import time
from pathlib import Path


class LegoDetector:
    """Detects Lego pieces using YOLO model"""
    
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5):
        """
        Initialize the YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detection
        """
        self.confidence_threshold = confidence_threshold
        self.detections = []
        self.last_detection_time = 0.0
        self.detection_interval = 5.0
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"[DETECTOR] Loaded YOLO model: {model_path}")
        except ImportError:
            print("WARNING: ultralytics not installed. Install with: pip install ultralytics")
            print("Using mock detector for testing")
            self.model = None
    
    def detect(self, frame):
        """
        Detect Lego pieces in a frame
        
        Args:
            frame: Input image frame (BGR)
            
        Returns:
            List of detections with format:
            [{"id": int, "x": float, "y": float, "width": float, "height": float, 
              "confidence": float, "class": str}, ...]
        """
        if self.model is None:
            detections = self._mock_detect(frame)
            self.detections = detections
            return detections

        now = time.time()
        if now - self.last_detection_time < self.detection_interval and self.detections:
            return self.detections

        # Run YOLO inference with default class labels
        results = self.model(frame, conf=self.confidence_threshold)
        result = results[0]
        
        detections = []
        for idx, box in enumerate(result.boxes):
            xyxy = np.asarray(box.xyxy).flatten()
            if xyxy.size != 4:
                continue
            x1, y1, x2, y2 = xyxy.tolist()
            
            conf = float(box.conf.cpu().numpy()) if hasattr(box, "conf") else float(box.conf)
            cls_id = int(box.cls.cpu().numpy()) if hasattr(box, "cls") else int(box.cls)
            class_name = self.model.names.get(cls_id, str(cls_id))
            
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            detections.append({
                "id": idx,
                "x": float(x),
                "y": float(y),
                "width": float(width),
                "height": float(height),
                "confidence": float(conf),
                "class": class_name
            })
        
        self.detections = detections
        self.last_detection_time = now
        return detections
    
    def _mock_detect(self, frame):
        """Mock detector for testing (returns dummy detections)"""
        return [
            {"id": 0, "x": 100, "y": 100, "width": 50, "height": 50, "confidence": 0.95, "class": "Lego"},
            {"id": 1, "x": 300, "y": 150, "width": 45, "height": 45, "confidence": 0.92, "class": "Lego"},
        ]
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes on frame for detected Legos
        
        Args:
            frame: Input image frame
            detections: List of detections
            
        Returns:
            Frame with drawn boxes and labels
        """
        annotated = frame.copy()
        
        for detection in detections:
            x = int(detection["x"])
            y = int(detection["y"])
            w = int(detection["width"] / 2)
            h = int(detection["height"] / 2)
            
            # Draw bounding box (green)
            cv2.rectangle(annotated, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center point (red)
            cv2.circle(annotated, (x, y), 5, (0, 0, 255), -1)
            
            # Draw label for detected object
            label = f"{detection['class']} {detection['id']} ({detection['confidence']:.2f})"
            cv2.putText(annotated, label, (x - w, y - h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated
    
    def get_detection_by_id(self, lego_id):
        """
        Get detection details by Lego ID
        
        Args:
            lego_id: The ID of the Lego to find
            
        Returns:
            Detection dictionary or None if not found
        """
        for detection in self.detections:
            if detection["id"] == lego_id:
                return detection
        return None
    
    def get_all_detections(self):
        """Get all current detections"""
        return self.detections.copy()
    
    def count_detections(self):
        """Get count of detected Lego pieces"""
        return len(self.detections)


class ColorBasedDetector:
    """Alternative detector using color-based detection for red Lego bricks"""
    
    def __init__(self, hue_range=(0, 10), saturation_range=(100, 255), value_range=(100, 255)):
        """
        Initialize color-based detector
        
        Args:
            hue_range: HSV hue range for red color
            saturation_range: HSV saturation range
            value_range: HSV value range
        """
        self.hue_range = hue_range
        self.saturation_range = saturation_range
        self.value_range = value_range
        self.detections = []
    
    def detect(self, frame):
        """
        Detect red Lego bricks using color-based thresholding
        
        Args:
            frame: Input BGR image
            
        Returns:
            List of detections
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for red color
        lower = np.array([self.hue_range[0], self.saturation_range[0], self.value_range[0]])
        upper = np.array([self.hue_range[1], self.saturation_range[1], self.value_range[1]])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    x = int(M["m10"] / M["m00"])
                    y = int(M["m01"] / M["m00"])
                    
                    x1, y1, w, h = cv2.boundingRect(contour)
                    
                    detections.append({
                        "id": idx,
                        "x": float(x),
                        "y": float(y),
                        "width": float(w),
                        "height": float(h),
                        "confidence": min(1.0, area / 1000.0),
                        "class": "Lego"
                    })
        
        self.detections = detections
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw detections on frame"""
        annotated = frame.copy()
        for detection in detections:
            x = int(detection["x"])
            y = int(detection["y"])
            w = int(detection["width"] / 2)
            h = int(detection["height"] / 2)
            
            cv2.rectangle(annotated, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(annotated, (x, y), 5, (0, 0, 255), -1)
        
        return annotated
