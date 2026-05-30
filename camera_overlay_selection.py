"""
camera_overlay_selection.py - Click-to-Grab
Listens for mouse clicks on the live video feed.
When clicking inside a detection box, grabs that Lego's location and sends to robot.
"""

import cv2
import numpy as np
import time
from PyQt5.QtCore import Qt

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
    
    def get_active_click_markers(self):
        """Return active click markers and drop expired markers"""
        now = time.time()
        self.click_markers = [m for m in self.click_markers if now - m["created"] < m["duration"]]
        return self.click_markers.copy()
    
    def clear_history(self):
        """Clear selection history"""
        self.selected_history.clear()
        self.click_markers.clear()
    
    def _add_click_marker(self, position, duration=5.0):
        """Add a temporary click marker for feedback"""
        self.click_markers.append({
            "position": (int(position[0]), int(position[1])),
            "created": time.time(),
            "duration": duration
        })


class OverlayRenderer:
    """Renders interactive overlay on camera feed"""
    
    def __init__(self):
        """Initialize overlay renderer"""
        self.highlight_color = (0, 255, 0)  # Green for normal
        self.hover_color = (255, 255, 0)    # Yellow for hover
        self.selected_color = (0, 0, 255)   # Red for selected
        self.line_thickness = 2
    
    def draw_interactive_boxes(self, frame, detections, hover_detection=None, selected_detection=None, click_markers=None):
        """
        Draw interactive bounding boxes on frame
        
        Args:
            frame: Camera frame
            detections: List of detections
            hover_detection: Detection under mouse hover
            selected_detection: Currently selected detection
            click_markers: List of temporary click markers
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for detection in detections:
            x = int(detection["x"])
            y = int(detection["y"])
            w = int(detection["width"] / 2)
            h = int(detection["height"] / 2)
            
            # Determine color based on state
            if selected_detection and detection["id"] == selected_detection["id"]:
                color = self.selected_color
                thickness = self.line_thickness + 2
            elif hover_detection and detection["id"] == hover_detection["id"]:
                color = self.hover_color
                thickness = self.line_thickness + 1
            else:
                color = self.highlight_color
                thickness = self.line_thickness
            
            # Draw box
            cv2.rectangle(annotated, (x - w, y - h), (x + w, y + h), color, thickness)
            
            # Draw center point
            cv2.circle(annotated, (x, y), 4, color, -1)
            
            # Draw label with clickable indicator
            label = f"{detection.get('class', 'Object')} {detection['id']} - Click to grab"
            text_x = x - w
            text_y = y - h - 5
            
            # Add background to text for readability
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            cv2.rectangle(annotated, 
                         (text_x - 2, text_y - text_size[1] - 2),
                         (text_x + text_size[0] + 2, text_y + 2),
                         color, -1)
            
            cv2.putText(annotated, label, (text_x, text_y),
                       font, font_scale, (0, 0, 0) if color == self.highlight_color else (0, 0, 0), 
                       font_thickness)

        if click_markers:
            for marker in click_markers:
                px, py = marker["position"]
                cv2.circle(annotated, (px, py), 10, (255, 0, 0), -1)
                cv2.circle(annotated, (px, py), 14, (255, 0, 0), 2)

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
        self.selected_detection = None
        self.hover_detection = None
    
    def update_detections(self, detections, frame):
        """Update current detections and frame"""
        self.click_handler.set_detections(detections, frame)
    
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
        click_markers = self.click_handler.get_active_click_markers()
        return self.overlay_renderer.draw_interactive_boxes(
            frame, detections, self.hover_detection, self.selected_detection, click_markers
        )
    
    def get_selected_object(self):
        """Get currently selected object"""
        return self.selected_detection
    
    def clear_selection(self):
        """Clear current selection"""
        self.selected_detection = None


class ClickableBox:
    """Represents a clickable bounding box region"""
    
    def __init__(self, x, y, width, height, lego_id, callback=None):
        """
        Initialize clickable box
        
        Args:
            x, y: Center coordinates
            width, height: Dimensions
            lego_id: ID of associated Lego
            callback: Function to call on click
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.lego_id = lego_id
        self.callback = callback
        self.is_hovered = False
    
    def contains_point(self, px, py):
        """Check if point is within box"""
        left = self.x - self.width / 2
        right = self.x + self.width / 2
        top = self.y - self.height / 2
        bottom = self.y + self.height / 2
        
        return left <= px <= right and top <= py <= bottom
    
    def on_click(self):
        """Handle click event"""
        if self.callback:
            self.callback(self.lego_id, (self.x, self.y))
    
    def draw(self, frame, color=(0, 255, 0), thickness=2):
        """Draw box on frame"""
        x = int(self.x)
        y = int(self.y)
        w = int(self.width / 2)
        h = int(self.height / 2)
        
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), color, thickness)
        return frame
