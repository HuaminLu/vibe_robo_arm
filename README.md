# Collaborative Robot Arm - Enterprise Fleet System Setup Guide

## Overview
This system implements an enterprise-grade multi-robot assembly line controller with real-time vision, safety monitoring, and intelligent task delegation.

## System Components

### Core Scripts

#### **main_enterprise.py** (Enterprise Andon Dashboard)
- Industrial-grade PyQt5 interface mimicking real factory software
- Displays live camera feed with real-time object detection
- **Andon Board**: Status indicators showing system health (GREEN/YELLOW/RED)
- **Robot Status Panel**: Individual status for each connected arm
- **Fleet Control**: Switch between Manual, Custom, and Automatic modes

#### **hand_control_fleet.py** (Fleet Commander)
- Manages multiple robot arms simultaneously
- Auto-detects available COM ports on startup
- Supports up to 10+ robot arms in parallel
- **RobotArm Class**: Individual arm control with queued commands
- **FleetController**: Coordinates across all connected arms
- **TaskDispatcher**: Intelligent task assignment strategies

#### **object_detection.py** (The Tracker)
- YOLO-based object detection for Lego pieces
- Color-based detection alternative (for red bricks)
- Real-time bounding box generation
- Confidence scoring for detection quality

#### **hand_safety.py** (The Lifeguard)
- Continuous background monitoring
- Skin color detection for human hands
- Configurable danger zones
- Instant emergency stop trigger
- Works across entire fleet

#### **camera_overlay_selection.py** (Click-to-Grab)
- Mouse click detection on video feed
- Single-robot task selection
- Visual feedback on selection

#### **camera_overlay_selection_fleet.py** (Click-to-Grab Fleet Edition)
- Enhanced multi-robot task dispatch
- Automatic arm assignment based on proximity
- Multiple assignment strategies (Closest, Idle-First, Load-Balanced)
- Adaptive routing with task learning

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Windows 10/11, Linux, or macOS
- USB camera (Orbbec or standard USB camera)
- Dobot robot arm(s) connected via USB

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install numpy opencv-python PyQt5 pyserial ultralytics
```

### Step 2: Download YOLO Model

The system will automatically download the YOLOv8 nano model on first run, or manually:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Step 3: Configure Robot COM Ports

The system auto-detects available COM ports. To manually verify:

**Windows:**
1. Connect robot arm via USB
2. Open Device Manager
3. Check COM port under "Ports (COM & LPT)"

**Linux:**
- Connected device appears as `/dev/ttyUSB0` or similar

**macOS:**
- Connected device appears as `/dev/tty.usbserial-*`

---

## Running the System

### Single Robot (Original System)
```bash
python main.py
```

### Multi-Robot Enterprise System
```bash
python main_enterprise.py
```

**On startup:**
- System automatically scans for COM ports
- Attempts to connect to available robot arms
- Displays connection status in the GUI
- Initializes Andon board with GREEN status

---

## Operating Modes

### 1. Manual Mode (Collaborative Handoff)
- **How it works:**
  1. Live camera feed shows detected Lego pieces with bounding boxes
  2. Click on any highlighted object
  3. System finds closest available robot arm
  4. Selected arm automatically picks up the object
  5. Brings it to the handoff zone near the operator

- **Use case:** Human-robot collaboration for selective pickup

### 2. Custom Mode (Assembly Line Pathing)
- **How it works:**
  1. Operator defines waypoints (A, B, C) via UI
  2. Each robot arm repeats the same path
  3. Arms work in parallel without collision
  4. Ideal for assembly line simulation

- **Use case:** Teaching specific paths; factory floor simulation

### 3. Automatic Mode (Swarm Sorting)
- **How it works:**
  1. System scans entire table for all objects
  2. Creates prioritized task list
  3. Distributes tasks across available arms
  4. Arms work simultaneously to clear workspace
  5. Continues until no objects remain

- **Use case:** High-speed autonomous collection

---

## Safety Features

### The Andon Board
Displays real-time system health:
- **● SYSTEM**: GREEN (running), YELLOW (waiting), RED (error)
- **● SAFETY**: GREEN (safe), RED (EMERGENCY STOP)
- **● MODE**: Current operating mode
- **● ACTIVE ARMS**: Number of connected robots

### Safety Override
At any time:
1. **EMERGENCY STOP button**: Instantly retracts ALL arms to safe position
2. **Hand detection**: If human hand detected, system triggers emergency stop
3. **Manual pause**: Pauses all operations; arms hold position

### Danger Zones
- Configurable circular safety zones around workspace
- Visual overlay on camera feed (GREEN = safe, RED = danger)
- Real-time hand detection within zones
- 5-second timeout before operations resume after hand clears

---

## GUI Layout

### Main Window
```
┌─────────────────────────────────┬──────────────────┐
│                                 │  Fleet Control   │
│      Andon Board (Status)        │  Panel           │
├─────────────────────────────────┤──────────────────┤
│                                 │  Mode Selection  │
│                                 │  ┌────────────┐  │
│   Live Camera Feed              │  │ Manual     │  │
│   with Object Boxes             │  │ Custom     │  │
│                                 │  │ Automatic  │  │
│                                 │  └────────────┘  │
│                                 │                  │
│                                 │  [Start]         │
│                                 │  [Emergency Stop]│
│                                 │  [Pause]         │
│                                 │                  │
│                                 │  Connected Robots│
│                                 │  ┌────────────┐  │
│                                 │  │ Arm 1 COM3 │  │
│                                 │  │ Status: ... │  │
│                                 │  │            │  │
│                                 │  │ Arm 2 COM4 │  │
│                                 │  │ Status: ... │  │
│                                 │  └────────────┘  │
└─────────────────────────────────┴──────────────────┘
```

---

## Task Dispatcher Strategies

### Closest
- Assigns task to spatially nearest available robot
- **Best for:** Minimizing movement time
- **Formula:** Euclidean distance from arm to target

### Idle-First
- Assigns to first robot that finishes its queue
- **Best for:** Balanced workload
- **Benefit:** Ensures all arms stay active

### Load-Balanced
- Assigns to arm with smallest command queue
- **Best for:** Optimal throughput
- **Metric:** Task queue length

---

## Configuration

### Robot Configuration
**File:** `hand_control_fleet.py`

```python
# Adjust robot parameters
arm.safe_height = 50       # Height when not picking
arm.pickup_height = 5      # Height when gripping
arm.drop_zone = (200, 100, 50)  # Drop-off location
arm.speed = 50             # Speed 0-100%
```

### Vision Configuration
**File:** `object_detection.py`

```python
# YOLO confidence threshold
detector = LegoDetector(confidence_threshold=0.5)  # 0-1

# Color-based detection range (for red bricks)
detector = ColorBasedDetector(
    hue_range=(0, 10),
    saturation_range=(100, 255),
    value_range=(100, 255)
)
```

### Safety Configuration
**File:** `hand_safety.py`

```python
safety = SafetyMonitor(
    robot_controller,
    danger_zone_radius=100,  # Pixels
    timeout=5                 # Seconds before resume
)
safety.set_danger_zone(center_x=150, center_y=150, radius=100)
```

---

## Troubleshooting

### Robot Not Detected
1. Check USB connections
2. Verify robot power is ON
3. Run calibration routine
4. Check Device Manager for COM port
5. Update COM port in code if needed

### Camera Feed Not Showing
1. Ensure camera is connected
2. Verify camera permissions
3. Test camera with: `python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"`
4. Try different camera index (0, 1, 2, ...)

### Objects Not Detected
1. Check lighting conditions
2. Adjust YOLO confidence threshold
3. Ensure YOLO model is downloaded
4. Try color-based detection for red bricks

### Emergency Stop Not Working
1. Check safety monitor thread is running
2. Verify robot connectivity
3. Test with `fleet_controller.emergency_stop_all()`

---

## Performance Tips

### Maximize Throughput
1. Use **Automatic Mode** for fastest collection
2. Set dispatcher to **"load_balanced"** strategy
3. Increase robot speed (up to 100%)
4. Add more robots if available

### Ensure Safety
1. Keep hands away from workspace
2. Use YELLOW zone warnings proactively
3. Test emergency stop regularly
4. Monitor Andon board continuously

### Optimize Accuracy
1. Adjust camera angle for better visibility
2. Improve lighting around objects
3. Lower YOLO confidence threshold (0.3-0.5)
4. Use color-based detection for known colors

---

## Advanced Features

### Task Learning
```python
router = AdaptiveTaskRouter(fleet_controller)
router.record_task(arm_id=1, location=(100, 100), completion_time=2.5)
best_arm = router.get_best_arm_for_location((100, 100))
```

### Custom ARM Protocols
Extend `RobotArm` class for different robot types:
```python
class CustomRobotArm(RobotArm):
    def _send_move_command(self, target, speed):
        # Implement custom protocol
        pass
```

### Real-time Monitoring
```python
status = fleet_controller.get_fleet_status()
for arm_id, status in status.items():
    print(f"Arm {arm_id}: {status['state']}")
```

---

## File Structure

```
vibe_robo_arm/
├── main.py                              # Single robot version
├── main_enterprise.py                   # Multi-robot version
├── object_detection.py                  # Vision module
├── hand_control.py                      # Single arm control
├── hand_control_fleet.py                # Multi-arm control
├── hand_safety.py                       # Safety monitoring
├── camera_overlay_selection.py          # Single robot click handler
├── camera_overlay_selection_fleet.py   # Multi-robot click handler
├── requirements.txt                     # Dependencies
└── README_ENTERPRISE.md                 # This file
```

---

## Support & Contributing

For issues, improvements, or custom robot integrations:
1. Check troubleshooting section above
2. Review robot-specific documentation
3. Test with mock mode first (robot_interface=None)

---

**Version:** 2.0 Enterprise Fleet Edition
**Last Updated:** May 2026
