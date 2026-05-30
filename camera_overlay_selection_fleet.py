"""
camera_overlay_selection_fleet.py - Click-to-Grab (Fleet Edition)
Enhanced version for multi-robot task delegation.
Listens for mouse clicks and intelligently dispatches to nearest available robot.
"""

import cv2
import numpy as np
from PyQt5.QtCore import Qt


class FleetClickHandler:
    """Handles mouse click events with fleet-aware task delegation"""
    
    def __init__(self, fleet_controller, on_selection_callback=None):
        """
        Initialize fleet-aware click handler
        
        Args:
            fleet_controller: Reference to FleetController for task dispatch
            on_selection_callback: Function to call when object selected
        """
        self.fleet_controller = fleet_controller
        self.on_selection_callback = on_selection_callback
        self.current_detections = []
        self.last_frame = None
        self.selected_history = []
        self.task_dispatcher = TaskDispatcher(fleet_controller)
    
    def set_detections(self, detections, frame):
        """Update current detections and frame"""
        self.current_detections = detections
        self.last_frame = frame
    
    def handle_click(self, click_x, click_y, display_width, display_height):
        """
        Handle mouse click on video display
        Automatically dispatches to closest available arm
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
            self._handle_fleet_selection(selected_detection)
        else:
            print(f"[CLICK] No object at ({frame_x}, {frame_y})")
    
    def _find_clicked_detection(self, click_x, click_y):
        """Find which detection was clicked"""
        for detection in self.current_detections:
            x = detection["x"]
            y = detection["y"]
            width = detection["width"]
            height = detection["height"]
            
            left = x - width / 2
            right = x + width / 2
            top = y - height / 2
            bottom = y + height / 2
            
            if left <= click_x <= right and top <= click_y <= bottom:
                return detection
        
        return None
    
    def _handle_fleet_selection(self, detection):
        """
        Handle selection with fleet dispatch
        Determines which robot is best suited for the task
        """
        lego_id = detection["id"]
        location = (detection["x"], detection["y"])
        confidence = detection["confidence"]
        
        print(f"[CLICK] Selected Lego {lego_id} at {location} (confidence: {confidence:.2f})")
        
        # Dispatch to nearest available arm
        assigned_arm = self.task_dispatcher.dispatch_detection(detection)
        
        if assigned_arm:
            print(f"[CLICK] Task dispatched to Arm {assigned_arm}")
        else:
            print("[CLICK] No available arms for task")
        
        # Record in history
        self.selected_history.append({
            "lego_id": lego_id,
            "location": location,
            "confidence": confidence,
            "assigned_arm": assigned_arm
        })
        
        # Call callback if provided
        if self.on_selection_callback:
            self.on_selection_callback(lego_id, location)
    
    def get_selection_history(self):
        """Get history of selections"""
        return self.selected_history.copy()
    
    def clear_history(self):
        """Clear selection history"""
        self.selected_history.clear()


class TaskDispatcher:
    """Intelligent task dispatcher for fleet"""
    
    def __init__(self, fleet_controller):
        """
        Initialize task dispatcher
        
        Args:
            fleet_controller: Reference to FleetController
        """
        self.fleet = fleet_controller
        self.assignment_strategy = "closest"
        self.priority_zones = {}
    
    def dispatch_detection(self, detection):
        """
        Dispatch detected object to best available arm
        
        Args:
            detection: Detection dictionary
            
        Returns:
            Arm ID if assigned, None otherwise
        """
        target_location = (detection["x"], detection["y"])
        
        # Select arm based on strategy
        if self.assignment_strategy == "closest":
            arm_id = self._assign_closest(target_location)
        elif self.assignment_strategy == "idle_first":
            arm_id = self._assign_idle_first()
        else:  # load_balanced
            arm_id = self._assign_load_balanced()
        
        if arm_id:
            self.fleet.dispatch_task(arm_id, target_location)
            return arm_id
        
        return None
    
    def _assign_closest(self, target_location):
        """Assign to closest available arm"""
        target_x, target_y = target_location
        closest_arm = None
        min_distance = float('inf')
        
        for arm_id, arm in self.fleet.arms.items():
            # Check if arm is available
            if not (arm.is_connected and arm.state.name == "IDLE"):
                continue
            
            arm_x, arm_y, _ = arm.current_position
            distance = ((arm_x - target_x)**2 + (arm_y - target_y)**2)**0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_arm = arm_id
        
        return closest_arm
    
    def _assign_idle_first(self):
        """Assign to first idle arm"""
        for arm_id, arm in self.fleet.arms.items():
            if arm.is_connected and arm.state.name == "IDLE":
                return arm_id
        return None
    
    def _assign_load_balanced(self):
        """Assign to arm with smallest queue"""
        min_queue_size = float('inf')
        best_arm = None
        
        for arm_id, arm in self.fleet.arms.items():
            if arm.is_connected:
                queue_size = len(arm.command_queue)
                if queue_size < min_queue_size:
                    min_queue_size = queue_size
                    best_arm = arm_id
        
        return best_arm
    
    def set_strategy(self, strategy):
        """Set task assignment strategy"""
        if strategy in ["closest", "idle_first", "load_balanced"]:
            self.assignment_strategy = strategy
            print(f"[DISPATCHER] Strategy changed to: {strategy}")


class FleetAwareOverlayRenderer:
    """Renders interactive overlay with fleet status indicators"""
    
    def __init__(self):
        """Initialize overlay renderer"""
        self.highlight_color = (0, 255, 0)
        self.hover_color = (255, 255, 0)
        self.selected_color = (0, 0, 255)
        self.line_thickness = 2
    
    def draw_interactive_boxes(self, frame, detections, fleet_status, hover_detection=None, selected_detection=None):
        """
        Draw interactive bounding boxes with fleet assignment indicators
        
        Args:
            frame: Camera frame
            detections: List of detections
            fleet_status: Status of all arms in fleet
            hover_detection: Detection under mouse
            selected_detection: Currently selected detection
            
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
            
            # Draw label
            label = f"Lego {detection['id']} - Click to grab"
            text_x = x - w
            text_y = y - h - 5
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            cv2.rectangle(annotated, 
                         (text_x - 2, text_y - text_size[1] - 2),
                         (text_x + text_size[0] + 2, text_y + 2),
                         color, -1)
            
            cv2.putText(annotated, label, (text_x, text_y),
                       font, font_scale, (0, 0, 0), font_thickness)
        
        return annotated
    
    def draw_fleet_status_overlay(self, frame, fleet_status):
        """
        Draw fleet status information on frame
        
        Args:
            frame: Camera frame
            fleet_status: Dictionary of arm statuses
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw fleet status in top-right corner
        y_offset = 20
        for arm_id, status in fleet_status.items():
            text = f"Arm {arm_id}: {status['state']}"
            color = (0, 255, 0) if status['connection'] == "Connected" else (0, 0, 255)
            
            cv2.putText(annotated, text, (frame.shape[1] - 200, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
        
        return annotated
    
    def draw_danger_zones(self, frame, danger_zones, hand_detected=False):
        """
        Draw danger zones on frame
        
        Args:
            frame: Camera frame
            danger_zones: List of (center_x, center_y, radius) tuples
            hand_detected: Whether a hand is currently detected
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for center_x, center_y, radius in danger_zones:
            zone_color = (0, 0, 255) if hand_detected else (0, 255, 0)
            cv2.circle(annotated, (int(center_x), int(center_y)), radius, zone_color, 2)
        
        return annotated


class AdaptiveTaskRouter:
    """Adaptive router that learns from task history"""
    
    def __init__(self, fleet_controller):
        """Initialize adaptive router"""
        self.fleet = fleet_controller
        self.task_history = []
        self.arm_performance = {}
        self.zone_preferences = {}
    
    def record_task(self, arm_id, location, completion_time):
        """Record completed task for learning"""
        self.task_history.append({
            "arm_id": arm_id,
            "location": location,
            "time": completion_time
        })
        
        # Update arm performance metrics
        if arm_id not in self.arm_performance:
            self.arm_performance[arm_id] = []
        self.arm_performance[arm_id].append(completion_time)
    
    def get_best_arm_for_location(self, location):
        """Get best arm for given location based on history"""
        x, y = location
        
        # Create zone key
        zone_x = int(x // 100)
        zone_y = int(y // 100)
        zone_key = (zone_x, zone_y)
        
        # Return preferred arm for this zone if exists
        if zone_key in self.zone_preferences:
            preferred_arm = self.zone_preferences[zone_key]
            if preferred_arm in self.fleet.arms:
                arm = self.fleet.arms[preferred_arm]
                if arm.is_connected and arm.state.name == "IDLE":
                    return preferred_arm
        
        # Default to closest arm
        target_x, target_y = location
        closest_arm = None
        min_distance = float('inf')
        
        for arm_id, arm in self.fleet.arms.items():
            if arm.is_connected and arm.state.name == "IDLE":
                arm_x, arm_y, _ = arm.current_position
                distance = ((arm_x - target_x)**2 + (arm_y - target_y)**2)**0.5
                
                if distance < min_distance:
                    min_distance = distance
                    closest_arm = arm_id
        
        # Update zone preference
        if closest_arm:
            self.zone_preferences[zone_key] = closest_arm
        
        return closest_arm
