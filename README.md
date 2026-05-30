### **How Everything Works Together (Architecture)**
To keep the video feed smooth and ensure safety stops happen instantly, the system runs three separate tasks simultaneously:

1. **The Interface (Frontend):** Runs the GUI, shows the live video, and registers button clicks.
2. **The Eyes & Safety (Vision):** Constantly analyzes the camera feed to find Lego pieces and watch for human hands.
3. **The Muscle (Control):** Manages the physical robot movements based on commands from the Interface or emergency stops from the Vision.

### **The Scripts**

- **`main.py` (The Hub):** The main graphical window (using PyQt). It displays the live camera feed with boxes drawn around Legos, houses the mode-selection buttons, and routes commands between the vision and control scripts.
- **`object_detection.py` (Lego Finder):** Uses a YOLO AI model to spot Lego pieces in the camera feed. It figures out exactly where they are on the table so the robot knows where to reach.
- **`hand_control.py` (Robot Driver):** Translates target coordinates into physical robot movements. It handles the specific sequence for picking up an object, moving it, and dropping it.
- **`hand_safety.py` (The Lifeguard):** Runs constantly in the background. It specifically looks for human hands entering a defined "danger zone." If it sees a hand, it immediately overrides all other scripts and triggers an emergency retract.
- **`camera_overlay_selection.py` (Click-to-Grab):** Listens for your mouse clicks on the live video feed. If you click inside a box drawn by the object detector, this script grabs that specific Lego's location and tells the robot to fetch it.

### **The Operating Modes**

- **Manual Mode:** You act as the dispatcher. You look at the live video feed, click on a highlighted Lego piece, and the robot automatically moves to grab it and places it in a designated drop-off zone (like a plate).
- **Custom Mode:** You teach the robot a specific path. You set points (A, B, C), and the robot continuously loops through this predetermined path, picking up any items it finds at the target location.
- **Automatic Mode:** The system takes full control. It scans the table, identifies all visible Lego pieces, and systematically sends the robot to collect every single one without any user input.

### **The Safety Procedure**
Regardless of which mode is active, **`hand_safety.py`** never stops running. If a human hand crosses into the operational space, the safety script instantly fires an interrupt to **`hand_control.py`**. The robot halts its current task, pulls up to a safe retracted position, and waits for the area to clear (e.g., a 5-second timeout) before resuming or waiting for a new command.



## Basic Robot Control
Refer to `testDobot.py` for basic Dobot control codes

**Notes**

- Do not modify the lib folder (unless you're sure of what you're doing) -> it includes the DLLs to interface with DobotLink
- dobotArm.py contains the python wrapper functions. Try to call from this library unless absolutely needed. Only modify the functions inside if you know what you are doing



### Using Intrinsic Calibration Data

You need to first import all the data

```python
data          = np.load("camera_params.npz")
camera_matrix = data["camera_matrix"]   # 3x3 intrinsic matrix  (K)
dist_coeffs   = data["dist_coeffs"]     # distortion vector
```

- Option A - undistort a single frame (simple):
  `undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)`

- Option B - fast per-frame using pre-computed maps (recommended for live video):

```python
h, w = frame.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), alpha=1)
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs,
                                              None, new_K, (w,h), cv2.CV_16SC2)
undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
```

- Option C - pose estimation with solvePnP:
  `cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)`


