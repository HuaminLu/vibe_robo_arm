"""
hand_control.py - Robot Driver
Translates target coordinates into physical robot movements.
Handles the sequence for picking up an object, moving it, and dropping it.
"""

import time
import threading
from enum import Enum
from collections import deque


class RobotState(Enum):
    """State machine states for robot operation"""
    IDLE = 0
    MOVING_TO_TARGET = 1
    APPROACHING_OBJECT = 2
    GRIPPING = 3
    LIFTING = 4
    MOVING_TO_DROP = 5
    DROPPING = 6
    RETURNING = 7
    EMERGENCY_RETRACT = 8
    PAUSED = 9


class RobotController:
    """Controls the physical robot arm movements"""
    
    def __init__(self, robot_interface=None):
        """
        Initialize robot controller
        
        Args:
            robot_interface: Interface to actual robot (Dobot, etc.)
                            If None, uses mock/simulator
        """
        self.robot_interface = robot_interface
        self.state = RobotState.IDLE
        self.running = True
        self.paused = False
        self.emergency_stop_flag = False
        
        # Configuration
        self.safe_height = 50  # Height when not picking
        self.pickup_height = 5  # Height when picking object
        self.drop_zone = (200, 100, 50)  # Default drop-off location (x, y, z)
        self.speed = 50  # Speed percentage (0-100)
        
        # State tracking
        self.current_position = (0, 0, 50)  # Current xyz position
        self.target_position = None
        self.is_gripping = False
        
        # Queue for commands
        self.command_queue = deque()
        
        # Start control loop in background
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
    
    def _control_loop(self):
        """Main control loop running in background thread"""
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            
            if self.emergency_stop_flag:
                self._execute_emergency_retract()
                time.sleep(0.1)
                continue
            
            # Process commands from queue
            if self.command_queue:
                command = self.command_queue.popleft()
                self._execute_command(command)
            
            time.sleep(0.05)
    
    def move_to_and_pick(self, target_location):
        """
        High-level command: Move to location and pick up object
        
        Args:
            target_location: (x, y) coordinates of target Lego piece
        """
        self.command_queue.append({
            "type": "pick",
            "location": target_location
        })
    
    def move_to_custom_points(self, waypoints):
        """
        Move through a series of waypoints (for Custom mode)
        
        Args:
            waypoints: List of (x, y) coordinates
        """
        self.command_queue.append({
            "type": "waypoints",
            "waypoints": waypoints
        })
    
    def start_automatic_mode(self):
        """Start automatic collection mode"""
        self.command_queue.append({
            "type": "automatic"
        })
    
    def pause(self):
        """Pause current operation"""
        self.paused = True
        self._move_to_safe_height()
    
    def resume(self):
        """Resume from pause"""
        self.paused = False
    
    def emergency_stop(self):
        """Trigger emergency stop - robot retracts immediately"""
        self.emergency_stop_flag = True
    
    def _execute_emergency_retract(self):
        """Emergency retraction sequence"""
        self.state = RobotState.EMERGENCY_RETRACT
        print("[ROBOT] EMERGENCY RETRACT INITIATED")
        
        # Open gripper if closed
        if self.is_gripping:
            self._control_gripper(False)
        
        # Move to safe position quickly
        self._move_to_position((0, 0, 100), speed=100)
        
        self.current_position = (0, 0, 100)
        self.state = RobotState.IDLE
        self.emergency_stop_flag = False
        print("[ROBOT] Emergency retract complete")
    
    def _execute_command(self, command):
        """Execute a queued command"""
        cmd_type = command.get("type")
        
        if cmd_type == "pick":
            self._execute_pick(command["location"])
        elif cmd_type == "waypoints":
            self._execute_waypoints(command["waypoints"])
        elif cmd_type == "automatic":
            self._execute_automatic()
    
    def _execute_pick(self, target_location):
        """Execute pick operation for a single object"""
        print(f"[ROBOT] Starting pick sequence for location {target_location}")
        self.state = RobotState.MOVING_TO_TARGET
        
        # Move to target location at safe height
        target_x, target_y = target_location
        approach_pos = (target_x, target_y, self.safe_height)
        self._move_to_position(approach_pos, speed=self.speed)
        
        # Approach object
        self.state = RobotState.APPROACHING_OBJECT
        pickup_pos = (target_x, target_y, self.pickup_height)
        self._move_to_position(pickup_pos, speed=self.speed // 2)
        
        # Grip object
        self.state = RobotState.GRIPPING
        self._control_gripper(True)
        time.sleep(0.5)
        
        # Lift object
        self.state = RobotState.LIFTING
        lift_pos = (target_x, target_y, self.safe_height)
        self._move_to_position(lift_pos, speed=self.speed // 2)
        
        # Move to drop zone
        self.state = RobotState.MOVING_TO_DROP
        drop_approach = (self.drop_zone[0], self.drop_zone[1], self.safe_height)
        self._move_to_position(drop_approach, speed=self.speed)
        
        # Approach drop zone
        self.state = RobotState.DROPPING
        self._move_to_position(self.drop_zone, speed=self.speed // 2)
        
        # Release object
        self._control_gripper(False)
        time.sleep(0.5)
        
        # Return to safe position
        self.state = RobotState.RETURNING
        self._move_to_position((0, 0, self.safe_height), speed=self.speed)
        
        self.state = RobotState.IDLE
        print("[ROBOT] Pick sequence complete")
    
    def _execute_waypoints(self, waypoints):
        """Execute movement through waypoints (Custom mode)"""
        print(f"[ROBOT] Starting waypoint sequence with {len(waypoints)} points")
        
        while self.running and not self.paused:
            for waypoint in waypoints:
                if self.paused or self.emergency_stop_flag:
                    break
                
                target_x, target_y = waypoint
                self._move_to_position((target_x, target_y, self.safe_height), speed=self.speed)
                time.sleep(0.5)  # Pause at each waypoint
        
        self.state = RobotState.IDLE
        print("[ROBOT] Waypoint sequence complete")
    
    def _execute_automatic(self):
        """Execute automatic collection mode"""
        print("[ROBOT] Automatic mode started - waiting for detections")
        # This will receive detection updates from main.py
        # Implementation depends on how detections are passed
        self.state = RobotState.IDLE
    
    def _move_to_position(self, target, speed=50):
        """
        Move robot to target position
        
        Args:
            target: (x, y, z) position
            speed: Speed percentage (0-100)
        """
        if self.robot_interface:
            # Send to actual robot
            self.robot_interface.move_to(target, speed)
        else:
            # Mock movement
            self._mock_move(target, speed)
        
        self.current_position = target
    
    def _move_to_safe_height(self):
        """Move to safe height at current x, y"""
        x, y, _ = self.current_position
        self._move_to_position((x, y, self.safe_height), speed=100)
    
    def _control_gripper(self, close=True):
        """
        Control gripper open/close
        
        Args:
            close: True to close, False to open
        """
        if self.robot_interface:
            self.robot_interface.set_gripper(close)
        else:
            state = "CLOSED" if close else "OPEN"
            print(f"[GRIPPER] {state}")
        
        self.is_gripping = close
    
    def _mock_move(self, target, speed):
        """Mock movement for testing"""
        duration = 1.0 * (100 - speed) / 100 + 0.2  # Simulate movement time
        print(f"[ROBOT] Moving to {target} at {speed}% speed (simulated {duration:.2f}s)")
        time.sleep(duration * 0.1)  # Reduced for testing
    
    def set_drop_zone(self, x, y, z=None):
        """
        Set the drop-off zone coordinates
        
        Args:
            x, y: Coordinates of drop zone
            z: Height (optional, defaults to safe_height)
        """
        if z is None:
            z = self.safe_height
        self.drop_zone = (x, y, z)
        print(f"[ROBOT] Drop zone set to {self.drop_zone}")
    
    def set_speed(self, speed):
        """
        Set movement speed (0-100)
        
        Args:
            speed: Speed percentage
        """
        self.speed = max(0, min(100, speed))
    
    def get_position(self):
        """Get current robot position"""
        return self.current_position
    
    def get_state(self):
        """Get current robot state"""
        return self.state
    
    def shutdown(self):
        """Shutdown robot controller"""
        self.running = False
        self.emergency_stop()
        if self.robot_interface:
            self.robot_interface.disconnect()
        print("[ROBOT] Shutting down")
