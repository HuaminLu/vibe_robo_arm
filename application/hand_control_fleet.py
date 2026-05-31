"""
hand_control_fleet.py - Fleet Commander
Manages multiple robot arms simultaneously with automatic COM port detection.
Coordinates task delegation across all connected robots.
"""

import threading
import time
import serial
from enum import Enum
from collections import defaultdict, deque
import sys
import serial.tools.list_ports


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
    DISCONNECTED = 10


class RobotArm:
    """Represents a single robot arm in the fleet"""
    
    def __init__(self, arm_id, com_port, baud_rate=115200):
        """
        Initialize robot arm
        
        Args:
            arm_id: Unique identifier for this arm
            com_port: COM port (e.g., "COM3")
            baud_rate: Serial communication baud rate
        """
        self.arm_id = arm_id
        self.com_port = com_port
        self.baud_rate = baud_rate
        
        self.state = RobotState.DISCONNECTED
        self.is_connected = False
        self.running = True
        self.paused = False
        
        # Position and configuration
        self.current_position = (0, 0, 50)
        self.safe_height = 50
        self.pickup_height = 5
        self.drop_zone = (200, 100, 50)
        self.speed = 50
        self.is_gripping = False
        
        # Task queue
        self.command_queue = deque()
        self.current_task = "Idle"
        
        # Serial connection
        self.serial_port = None
        
        # Start control thread
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
    
    def connect(self):
        """Attempt to connect to robot arm"""
        try:
            self.serial_port = serial.Serial(self.com_port, self.baud_rate, timeout=1)
            self.is_connected = True
            self.state = RobotState.IDLE
            print(f"[ARM {self.arm_id}] Connected on {self.com_port}")
            return True
        except Exception as e:
            print(f"[ARM {self.arm_id}] Failed to connect: {e}")
            self.is_connected = False
            self.state = RobotState.DISCONNECTED
            return False
    
    def disconnect(self):
        """Disconnect from robot arm"""
        if self.serial_port:
            self.serial_port.close()
        self.is_connected = False
        self.state = RobotState.DISCONNECTED
        print(f"[ARM {self.arm_id}] Disconnected from {self.com_port}")
    
    def _control_loop(self):
        """Main control loop for this arm"""
        while self.running:
            if not self.is_connected:
                time.sleep(0.5)
                continue
            
            if self.paused:
                time.sleep(0.1)
                continue
            
            # Process command queue
            if self.command_queue:
                command = self.command_queue.popleft()
                self._execute_command(command)
            
            time.sleep(0.05)
    
    def queue_command(self, command):
        """Queue a command for execution"""
        self.command_queue.append(command)
    
    def move_to_and_pick(self, target_location):
        """Queue pick-and-place operation"""
        self.queue_command({
            "type": "pick",
            "location": target_location
        })
        self.current_task = f"Picking from {target_location}"
    
    def emergency_stop(self):
        """Emergency retract"""
        self.state = RobotState.EMERGENCY_RETRACT
        self.command_queue.clear()
        self.queue_command({
            "type": "emergency_retract"
        })
    
    def pause(self):
        """Pause operations"""
        self.paused = True
        self._move_to_safe_height()
        self.current_task = "Paused"
    
    def resume(self):
        """Resume operations"""
        self.paused = False
    
    def _execute_command(self, command):
        """Execute a queued command"""
        cmd_type = command.get("type")
        
        if cmd_type == "pick":
            self._execute_pick(command["location"])
        elif cmd_type == "emergency_retract":
            self._execute_emergency_retract()
    
    def _execute_pick(self, target_location):
        """Execute pick operation"""
        print(f"[ARM {self.arm_id}] Starting pick at {target_location}")
        self.state = RobotState.MOVING_TO_TARGET
        
        target_x, target_y = target_location
        approach_pos = (target_x, target_y, self.safe_height)
        self._move_to_position(approach_pos)
        
        self.state = RobotState.APPROACHING_OBJECT
        pickup_pos = (target_x, target_y, self.pickup_height)
        self._move_to_position(pickup_pos)
        
        self.state = RobotState.GRIPPING
        self._control_gripper(True)
        time.sleep(0.5)
        
        self.state = RobotState.LIFTING
        lift_pos = (target_x, target_y, self.safe_height)
        self._move_to_position(lift_pos)
        
        self.state = RobotState.MOVING_TO_DROP
        drop_approach = (self.drop_zone[0], self.drop_zone[1], self.safe_height)
        self._move_to_position(drop_approach)
        
        self.state = RobotState.DROPPING
        self._move_to_position(self.drop_zone)
        
        self._control_gripper(False)
        time.sleep(0.5)
        
        self.state = RobotState.RETURNING
        self._move_to_position((0, 0, self.safe_height))
        
        self.state = RobotState.IDLE
        self.current_task = "Idle"
        print(f"[ARM {self.arm_id}] Pick complete")
    
    def _execute_emergency_retract(self):
        """Execute emergency retraction"""
        print(f"[ARM {self.arm_id}] Emergency retract")
        if self.is_gripping:
            self._control_gripper(False)
        self._move_to_position((0, 0, 100), speed=100)
        self.state = RobotState.IDLE
        self.current_task = "Idle"
    
    def _move_to_position(self, target, speed=None):
        """Move to target position"""
        if speed is None:
            speed = self.speed
        
        if self.is_connected:
            self._send_move_command(target, speed)
        else:
            self._mock_move(target, speed)
        
        self.current_position = target
    
    def _move_to_safe_height(self):
        """Move to safe height"""
        x, y, _ = self.current_position
        self._move_to_position((x, y, self.safe_height))
    
    def _control_gripper(self, close=True):
        """Control gripper"""
        if self.is_connected:
            self._send_gripper_command(close)
        else:
            state = "CLOSED" if close else "OPEN"
            print(f"[ARM {self.arm_id}] Gripper: {state}")
        self.is_gripping = close
    
    def _send_move_command(self, target, speed):
        """Send movement command to robot (depends on robot type)"""
        # This would be implemented based on the specific robot protocol
        print(f"[ARM {self.arm_id}] Move to {target} at {speed}%")
    
    def _send_gripper_command(self, close):
        """Send gripper command to robot"""
        # This would be implemented based on the specific robot protocol
        state = "CLOSE" if close else "OPEN"
        print(f"[ARM {self.arm_id}] Gripper {state}")
    
    def _mock_move(self, target, speed):
        """Mock movement for testing"""
        duration = 1.0 * (100 - speed) / 100 + 0.2
        print(f"[ARM {self.arm_id}] Moving to {target} at {speed}% (simulated {duration:.2f}s)")
        time.sleep(duration * 0.05)
    
    def get_status(self):
        """Get current arm status"""
        return {
            "arm_id": self.arm_id,
            "com_port": self.com_port,
            "state": self.state.name,
            "connected": self.is_connected,
            "position": self.current_position,
            "task": self.current_task,
            "is_busy": self.state != RobotState.IDLE
        }
    
    def shutdown(self):
        """Shutdown arm"""
        self.running = False
        self.disconnect()


class FleetController:
    """Controls a fleet of robot arms"""
    
    def __init__(self):
        """Initialize fleet controller"""
        self.arms = {}
        self.running = True
        self.paused = False
        self.emergency_stop_flag = False
        
        print(f"[FLEET] Fleet initialized with {len(self.arms)} arms")
    
    def _discover_and_connect_robots(self):
        """Auto-discover available COM ports and connect to robots"""
        try:
            available_ports = self._get_available_com_ports()
            print(f"[FLEET] Found {len(available_ports)} available COM ports")
            
            arm_id = 1
            for com_port in available_ports:
                # Attempt to connect
                arm = RobotArm(arm_id, com_port)
                if arm.connect():
                    self.arms[arm_id] = arm
                    arm_id += 1
                else:
                    arm.shutdown()
        
        except Exception as e:
            print(f"[FLEET] Error discovering robots: {e}")
    
    def _get_available_com_ports(self):
        """Get list of available COM ports"""
        ports = []
        try:
            for port_info in serial.tools.list_ports.comports():
                ports.append(port_info.device)
        except:
            # Fallback for Windows if pyserial doesn't detect ports
            if sys.platform == "win32":
                for i in range(1, 10):
                    ports.append(f"COM{i}")
        
        return ports
    
    def add_arm(self, com_port):
        """Add a single arm by COM port and attempt connection"""
        arm_id = max(self.arms.keys(), default=0) + 1
        arm = RobotArm(arm_id, com_port)
        if arm.connect():
            self.arms[arm_id] = arm
            print(f"[FLEET] Added arm {arm_id} on {com_port}")
            return True
        arm.shutdown()
        return False
    
    def get_closest_available_arm(self, target_location):
        """Find closest available arm to target"""
        target_x, target_y = target_location
        closest_arm = None
        min_distance = float('inf')
        
        for arm_id, arm in self.arms.items():
            if arm.is_connected and arm.state == RobotState.IDLE:
                arm_x, arm_y, _ = arm.current_position
                distance = ((arm_x - target_x)**2 + (arm_y - target_y)**2)**0.5
                
                if distance < min_distance:
                    min_distance = distance
                    closest_arm = arm_id
        
        return closest_arm
    
    def dispatch_task(self, arm_id, target_location):
        """Dispatch task to specific arm"""
        if arm_id in self.arms:
            self.arms[arm_id].move_to_and_pick(target_location)
            return True
        return False
    
    def start_automatic_mode(self):
        """Start automatic collection mode for all arms"""
        print("[FLEET] Automatic mode started")
        for arm in self.arms.values():
            if arm.is_connected:
                print(f"[FLEET] Arm {arm.arm_id} ready for automatic collection")
    
    def pause_all(self):
        """Pause all arms"""
        self.paused = True
        for arm in self.arms.values():
            arm.pause()
        print("[FLEET] All arms paused")
    
    def resume_all(self):
        """Resume all arms"""
        self.paused = False
        for arm in self.arms.values():
            arm.resume()
        print("[FLEET] All arms resumed")
    
    def emergency_stop_all(self):
        """Emergency stop all arms"""
        self.emergency_stop_flag = True
        for arm in self.arms.values():
            arm.emergency_stop()
        print("[FLEET] EMERGENCY STOP - All arms retracting")
    
    def get_fleet_status(self):
        """Get status of all arms"""
        status = {}
        for arm_id, arm in self.arms.items():
            status[arm_id] = {
                "arm_id": arm_id,
                "com_port": arm.com_port,
                "state": arm.state.name,
                "connection": "Connected" if arm.is_connected else "Disconnected",
                "task": arm.current_task,
                "position": arm.current_position
            }
        return status
    
    def shutdown(self):
        """Shutdown entire fleet"""
        self.running = False
        print("[FLEET] Shutting down all arms...")
        for arm in self.arms.values():
            arm.shutdown()
        print("[FLEET] Fleet shutdown complete")


class TaskDispatcher:
    """Intelligent task dispatcher for the fleet"""
    
    def __init__(self, fleet_controller):
        """
        Initialize task dispatcher
        
        Args:
            fleet_controller: Reference to FleetController
        """
        self.fleet = fleet_controller
        self.task_queue = deque()
        self.assignment_strategy = "closest"  # Options: "closest", "idle_first", "load_balanced"
    
    def dispatch_detection(self, detection):
        """
        Dispatch a detected object to an available arm
        
        Args:
            detection: Detection dictionary with location
        """
        target_location = (detection["x"], detection["y"])
        
        if self.assignment_strategy == "closest":
            arm_id = self.fleet.get_closest_available_arm(target_location)
        elif self.assignment_strategy == "idle_first":
            arm_id = self._get_idle_arm()
        else:  # load_balanced
            arm_id = self._get_least_loaded_arm()
        
        if arm_id:
            self.fleet.dispatch_task(arm_id, target_location)
            print(f"[DISPATCHER] Task assigned to Arm {arm_id}: {detection}")
    
    def _get_idle_arm(self):
        """Get first idle arm"""
        for arm_id, arm in self.fleet.arms.items():
            if arm.is_connected and arm.state == RobotState.IDLE:
                return arm_id
        return None
    
    def _get_least_loaded_arm(self):
        """Get arm with smallest queue"""
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

    def add_arm(self, com_port):
        """Initializes a new serial connection and adds the arm to the fleet"""
        try:
            # Example initialization (depends on your specific repo's syntax)
            new_robot = RobotArm(port=com_port, baudrate=115200) 
            arm_id = len(self.fleet) + 1
            self.fleet[arm_id] = {
                "robot": new_robot,
                "com_port": com_port,
                "state": "Idle",
                "task": "Waiting",
                "connection": "Connected"
            }
            return True
        except Exception as e:
            print(f"Failed to connect to {com_port}: {e}")
            return False
