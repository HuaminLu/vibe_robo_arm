import dobotArm
import lib.DobotDllType as dType
import numpy as np
import cv2
import time
import threading
import os
import contextlib

# Suppress standard Python/C++ environment logs
os.environ['GLOG_minloglevel'] = '2'

# --- MEDIAPIPE SAFETY FRAMEWORK IMPORTS ---
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

"""CONSTANTS"""
Z_SAFE = 40          
Z_PICK = 10        
Z_PICK_LOWER = -35   
STABILITY_LIMIT = 60 
PIXEL_TOLERANCE = 10 
TARGET_MIN_AREA = 20  
TARGET_MAX_AREA = 4000 
HAND_CLEAR_FRAMES = 30 
HAND_DETECTION_ENABLED = True 

# --- CUSTOM RETRACTION COORDINATES ---
CLEAR_X = 200.0
CLEAR_Y = -150.0
CLEAR_Z = 60.0
CLEAR_R = 0.0

# Shared Global Thread Communication Variables
machine_state = "paused" 
HAND_IN_WORKSPACE = False
current_object_idx = 0
total_objects_count = 0
cleared_frames_global = 0  # Share clearance status safely with the UI thread

# --- INITIALIZATION FOR CAMERA AND GEOMETRY ---
api = dType.load()
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("[WARN] Camera index 1 failed, trying index 0...")
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Unable to open camera. Check connection.")
    exit(1)

# Set resolution properties to exactly match your 640x480 calibration matrix
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- CONFIGURING THE MEDIAPIPE HAND LANDMARKER TASK ---
model_path = 'hand_landmarker.task'  
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.8,
    running_mode=vision.RunningMode.IMAGE 
)

print("[SYSTEM] Initializing MediaPipe engine framework...")
with open(os.devnull, 'w') as devnull:
    with contextlib.redirect_stderr(devnull):
        detector = vision.HandLandmarker.create_from_options(options)
print("[SYSTEM] MediaPipe engine successfully locked down.")

# Load calibration targets
H_matrix = np.load("HomographyMatrix.npy")
data = np.load("./camera_params.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs   = data["dist_coeffs"]

# Compute undistort maps once
ret, frame = cap.read()
if frame is None:
    print("[CRITICAL ERROR] Could not read initial frame from camera source.")
    exit(1)
h, w = frame.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1)
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2)

# Create the persistent unified window
WINDOW_NAME = "Robot System Feed"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

latest_processed_frame = frame.copy()
frame_lock = threading.Lock()


def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]

def robot_to_pixel(x, y, H):
    H_inv = np.linalg.inv(H)
    p = np.array([x, y, 1])
    uv = H_inv @ p
    uv /= uv[2]
    return int(round(uv[0])), int(round(uv[1]))

def get_workspace_boundary_area():
    """Defines the operational boundaries of the robot workspace in millimeters (mm)."""
    workspace_limits = {
        "X_MIN": 150.0,  
        "X_MAX": 320.0,  
        "Y_MIN": -180.0, 
        "Y_MAX": 220.0   
    }
    return workspace_limits

def get_hand_boundary_area():
    """Defines the wider, early-warning outer envelope for hand safe tripping (mm)."""
    hand_limits = {
        "X_MIN": 120.0,  
        "X_MAX": 360.0,  
        "Y_MIN": -240.0, 
        "Y_MAX": 240.0   
    }
    return hand_limits

def is_coordinate_in_range(pixel_u, pixel_v, H):
    robot_x, robot_y = pixel_to_robot(pixel_u, pixel_v, H)
    # Check against your custom wide hand limits for safety trips
    bounds = get_hand_boundary_area()
    if (bounds["X_MIN"] <= robot_x <= bounds["X_MAX"]) and \
       (bounds["Y_MIN"] <= robot_y <= bounds["Y_MAX"]):
        return True
    return False

def move_safe_descend(api, x, y, z, rHead=0):
    dobotArm.move_to_xyz(api, x, y, Z_SAFE, rHead)
    dobotArm.move_to_xyz(api, x, y, z, rHead)

def move_safe_ascend(api, x, y, rHead=0):
    dobotArm.move_to_xyz(api, x, y, Z_SAFE, rHead)

def move_between_points(api, start, end, rHead=0):
    dobotArm.move_to_xyz(api, start[0], start[1], Z_SAFE, rHead)
    dobotArm.move_to_xyz(api, end[0], end[1], Z_SAFE, rHead)


# --- BACKGROUND SAFETY MONITOR WORKER THREAD ---
def background_safety_monitor_worker(api):
    global HAND_IN_WORKSPACE, latest_processed_frame, cleared_frames_global
    cleared_frames = 0
    hazard_frames = 0  # NEW: Debounce counter for incoming threats
    
    print("[SYSTEM] Background MediaPipe Safety Worker Thread Launched with Thread Debouncing.")
    
    while cap.isOpened():
        ret, f = cap.read()
        if not ret or f is None:
            time.sleep(0.01)
            continue
            
        f = cv2.remap(f, map1, map2, cv2.INTER_LINEAR)
        h_f, w_f = f.shape[:2] 
        
        with frame_lock:
            latest_processed_frame = f.copy()
        
        rgb_frame = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_stderr = os.dup(2)
            os.dup2(devnull, 2)
            detection_result = detector.detect(mp_image)
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
            os.close(devnull)
        except Exception:
            continue

        try:
            pose = dType.GetPose(api)
            current_robot_x = pose[0]
            current_robot_y = pose[1]
        except Exception:
            current_robot_x, current_robot_y = 0.0, 0.0

        hand_is_actually_hazardous = False
        if HAND_DETECTION_ENABLED and detection_result.hand_landmarks:
            for landmark in detection_result.hand_landmarks[0]:
                pixel_u = int(landmark.x * w_f)
                pixel_v = int(landmark.y * h_f)
                
                rx, ry = pixel_to_robot(pixel_u, pixel_v, H_matrix)
                
                if is_coordinate_in_range(pixel_u, pixel_v, H_matrix):
                    if machine_state == "pick place":
                        distance_to_arm = np.sqrt((rx - current_robot_x)**2 + (ry - current_robot_y)**2)
                        if distance_to_arm < 85.0: # Expanded padding slightly
                            continue 
                    
                    hand_is_actually_hazardous = True
                    break

        # --- DEBUNCED SAFETY EVALUATION LINE ---
        if hand_is_actually_hazardous:
            hazard_frames += 1
            cleared_frames = 0
            
            # Require 3 consecutive frames of confirmed threat before killing the hardware queue
            if hazard_frames >= 3:
                if not HAND_IN_WORKSPACE:
                    print("\n[ALERT] True threat validated. Locking down hardware queues.")
                    HAND_IN_WORKSPACE = True
                    try:
                        dType.SetQueuedCmdForceStopExec(api) 
                        dType.SetQueuedCmdClear(api)
                    except Exception:
                        pass
        else:
            hazard_frames = 0
            cleared_frames += 1
            if cleared_frames >= HAND_CLEAR_FRAMES:
                HAND_IN_WORKSPACE = False
        
        cleared_frames_global = cleared_frames
        time.sleep(0.02)


# --- UPDATED MOVEMENT WRAPPERS WITH QUEUE START SIGNALLING ---
def move_safe_descend(api, x, y, z, rHead=0):
    # Append points to queue
    dobotArm.move_to_xyz(api, x, y, Z_SAFE, rHead)
    dobotArm.move_to_xyz(api, x, y, z, rHead)
    # CRITICAL: Force the queue engine to resume processing instructions
    dType.SetQueuedCmdStartExec(api)

def move_safe_ascend(api, x, y, rHead=0):
    dobotArm.move_to_xyz(api, x, y, Z_SAFE, rHead)
    dType.SetQueuedCmdStartExec(api)

def move_between_points(api, start, end, rHead=0):
    dobotArm.move_to_xyz(api, start[0], start[1], Z_SAFE, rHead)
    dobotArm.move_to_xyz(api, end[0], end[1], Z_SAFE, rHead)
    dType.SetQueuedCmdStartExec(api)
def draw_visual_overlays(base_frame):
    """Unified single-point rendering engine executed exclusively by the main thread."""
    h_f, w_f = base_frame.shape[:2]
    
    # Draw Inner Object Target Area (Green)
    bounds_robot = get_workspace_boundary_area()
    r1 = robot_to_pixel(bounds_robot["X_MIN"], bounds_robot["Y_MIN"], H_matrix)
    r2 = robot_to_pixel(bounds_robot["X_MAX"], bounds_robot["Y_MIN"], H_matrix)
    r3 = robot_to_pixel(bounds_robot["X_MAX"], bounds_robot["Y_MAX"], H_matrix)
    r4 = robot_to_pixel(bounds_robot["X_MIN"], bounds_robot["Y_MAX"], H_matrix)
    pts_robot = np.array([r1, r2, r3, r4], np.int32).reshape((-1, 1, 2))
    cv2.polylines(base_frame, [pts_robot], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # Draw Wider Hand Safety Shield Envelope (Magenta)
    bounds_hand = get_hand_boundary_area()
    h1 = robot_to_pixel(bounds_hand["X_MIN"], bounds_hand["Y_MIN"], H_matrix)
    h2 = robot_to_pixel(bounds_hand["X_MAX"], bounds_hand["Y_MIN"], H_matrix)
    h3 = robot_to_pixel(bounds_hand["X_MAX"], bounds_hand["Y_MAX"], H_matrix)
    h4 = robot_to_pixel(bounds_hand["X_MIN"], bounds_hand["Y_MAX"], H_matrix)
    pts_hand = np.array([h1, h2, h3, h4], np.int32).reshape((-1, 1, 2))
    cv2.polylines(base_frame, [pts_hand], isClosed=True, color=(255, 0, 255), thickness=1)

    if HAND_IN_WORKSPACE:
        cv2.rectangle(base_frame, (10, 20), (w_f - 10, 90), (0, 0, 255), cv2.FILLED)
        cv2.putText(base_frame, "!! EMERGENCY FREEZE: HAND IN WORKSPACE !!", (25, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    elif machine_state == "pick place":
        cv2.rectangle(base_frame, (10, 20), (440, 90), (0, 0, 0), cv2.FILLED)
        cv2.putText(base_frame, f"RUNNING: Object {current_object_idx} of {total_objects_count}", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(base_frame, f"Status: Safe ({cleared_frames_global}/{HAND_CLEAR_FRAMES} frames clear)", 
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return base_frame


def ensure_no_hand_or_pause(api):
    global machine_state
    if HAND_IN_WORKSPACE:
        print("\n[SAFETY INTERCEPT] Threat signature detected! Freezing robot hardware immediately...")
        try:
            dType.SetQueuedCmdForceStopExec(api)
            dType.SetQueuedCmdClear(api)
        except Exception as e:
            print(f"[SAFETY ERROR] Failed to freeze queue: {e}")
            
        # The main loop renders the freeze screen natively, ensuring OpenCV UI stability
        while HAND_IN_WORKSPACE:
            with frame_lock:
                f = latest_processed_frame.copy()
            f = draw_visual_overlays(f)
            cv2.imshow(WINDOW_NAME, f)
            cv2.waitKey(20)
            
        print("[SAFETY INTERCEPT] Hand cleared from workspace. Executing safe post-freeze exit sequence...")
        try:
            dType.SetQueuedCmdStartExec(api)
            time.sleep(0.1) 
            
            print("[SAFETY] Opening gripper to drop active payload...")
            dType.SetEndEffectorGripper(api, True, False, isQueued=0)
            time.sleep(0.5) 
            dType.SetEndEffectorGripper(api, False, False, isQueued=0) 
            
            print(f"[SAFETY] Moving to out-of-view clear position: X={CLEAR_X}, Y={CLEAR_Y}...")
            current_index = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, CLEAR_X, CLEAR_Y, CLEAR_Z, CLEAR_R, isQueued=0)[0]
            
            start_timeout = time.time()
            while True:
                if time.time() - start_timeout > 4.0:
                    print("[SAFETY WARN] Retraction movement timeout reached.")
                    break
                if dType.GetQueuedCmdCurrentIndex(api)[0] >= current_index:
                    break
                time.sleep(0.05)
            
        except Exception as e:
            print(f"[SAFETY ERROR] Recovery move failed: {e}")
            
        print("[SAFETY SYSTEM] Workspace clear. Raising break event redirection.")
        raise InterruptedError("Automation track safely broken due to workspace intrusion.")


def safe_sleep_with_monitoring(api, duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        ensure_no_hand_or_pause(api)
        
        # Keep OpenCV window responsive from the main thread during steps
        with frame_lock:
            f = latest_processed_frame.copy()
        f = draw_visual_overlays(f)
        cv2.imshow(WINDOW_NAME, f)
        cv2.waitKey(5)


def next_state():
    global machine_state
    if machine_state == "scanning plate":
        machine_state = "scanning target"
    elif machine_state == "scanning target":
        machine_state = "pick place"
    elif machine_state == "pick place":
        machine_state = "scanning plate"


def wait_for_space_to_restart():
    print("[INFO] Run complete. Press SPACE to scan again, or Q to quit.")
    while True:
        with frame_lock:
            f = latest_processed_frame.copy()
        f = draw_visual_overlays(f)
        cv2.imshow(WINDOW_NAME, f)
        key = cv2.waitKey(30) & 0xFF
        if key == 32:  
            return True
        if key == ord('q'):
            return False


def phase_detect_plates():
    global machine_state
    machine_state = "scanning plate"
    print("\n[PHASE 1] Scanning for drop zones. Waiting for stability...")
    stability_counter = 0
    last_count = 0
    
    while True:
        ensure_no_hand_or_pause(api)  
        with frame_lock:
            frame = latest_processed_frame.copy()
            
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)
        
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=100, 
            param1=110,      
            param2=35,      
            minRadius=15,   
            maxRadius=60    
        )

        current_list = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                if is_coordinate_in_range(i[0], i[1], H_matrix):
                    cv2.circle(display_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    rx, ry = pixel_to_robot(i[0], i[1], H_matrix)
                    current_list.append((rx, ry))

        if len(current_list) > 0 and len(current_list) == last_count:
            stability_counter += 1
        else:
            stability_counter = 0
            last_count = len(current_list)

        display_frame = draw_visual_overlays(display_frame)

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        cv2.putText(display_frame, f"LOCKING PLATES: {progress}% ({len(current_list)} found)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow(WINDOW_NAME, display_frame)
        cv2.waitKey(1)

        if stability_counter >= STABILITY_LIMIT:
            print(f"Locked {len(current_list)} plates.")
            return current_list


def phase_detect_targets(drop_list):
    global machine_state
    machine_state = "scanning target"
    print("\n[PHASE 2] Scanning for targets. Waiting for stability...")
    EXCLUSION_RADIUS_MM = 50.0  
    stability_counter = 0
    last_count = -1
    
    while True:
        ensure_no_hand_or_pause(api)
        with frame_lock:
            frame = latest_processed_frame.copy()
            
        display_frame = frame.copy()
        overlay_frame = np.zeros_like(frame)
        
        blurred_hsv = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv = cv2.cvtColor(blurred_hsv, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 120, 100]);   upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 120, 100]); upper_red2 = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        hsv_red = cv2.bitwise_or(mask1, mask2)
        
        red_channel = frame[:, :, 2].astype(np.int16)
        green_channel = frame[:, :, 1].astype(np.int16)
        blue_channel = frame[:, :, 0].astype(np.int16)
        red_dom = (red_channel > 90) & (red_channel > green_channel + 50) & (red_channel > blue_channel + 50)
        red_dom_mask = (red_dom.astype(np.uint8) * 255)
        combined = cv2.bitwise_and(hsv_red, red_dom_mask)
        mask = hsv_red if cv2.countNonZero(combined) == 0 else combined
        
        kernel_small = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_list = []
        print_strings = []
        
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if TARGET_MIN_AREA < area < TARGET_MAX_AREA:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    
                    if not is_coordinate_in_range(cx, cy, H_matrix):
                        continue
                        
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = w / float(h) if h != 0 else 0
                    fill_ratio = area / float(w * h) if w * h != 0 else 0
                    
                    if 0.15 < aspect_ratio < 6.0 and fill_ratio > 0.15:
                        rx, ry = pixel_to_robot(cx, cy, H_matrix)
                        
                        is_inside_plate = False
                        if drop_list is not None:
                            for drop_x, drop_y in drop_list:
                                distance = np.sqrt((rx - drop_x)**2 + (ry - drop_y)**2)
                                if distance < EXCLUSION_RADIUS_MM:
                                    is_inside_plate = True
                                    break
                        
                        if is_inside_plate:
                            cv2.circle(overlay_frame, (cx, cy), 6, (0, 0, 255), -1)
                            cv2.putText(overlay_frame, "EXCLUDED", (cx + 10, cy - 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                            continue 

                        rect = cv2.minAreaRect(cnt)
                        (cx_box, cy_box), (box_w, box_h), angle = rect

                        if box_w < box_h:
                            absolute_angle = angle + 90
                        else:
                            absolute_angle = angle

                        absolute_angle = absolute_angle % 180
                        grasp_angle = (180 - absolute_angle) % 180 
                        pick_r = int(grasp_angle)
                        
                        current_list.append((rx, ry, pick_r))

                        box = cv2.boxPoints(rect)
                        box = np.int64(box)
                        cv2.drawContours(overlay_frame, [box], 0, (255, 255, 0), 2)  
                        cv2.drawContours(overlay_frame, [cnt], 0, (0, 255, 0), 1)    
                        cv2.putText(overlay_frame, f"Obj {len(current_list)}: {pick_r}deg", (cx + 10, cy - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        print_strings.append(f"  -> Object {len(current_list)} detected at ({rx:.1f}, {ry:.1f}) | Target Angle: {pick_r}°")

        display_frame = cv2.addWeighted(display_frame, 1.0, overlay_frame, 1.0, 0)

        if drop_list is not None:
            for drop_x, drop_y in drop_list:
                p_u, p_v = robot_to_pixel(drop_x, drop_y, H_matrix)
                pixel_radius = int(EXCLUSION_RADIUS_MM * 1.5) 
                cv2.circle(display_frame, (p_u, p_v), pixel_radius, (255, 255, 0), 1, lineType=cv2.LINE_AA)

        if len(current_list) > 0 and len(current_list) == last_count:
            stability_counter += 1
        else:
            stability_counter = 0
            last_count = len(current_list)

        display_frame = draw_visual_overlays(display_frame)

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        cv2.rectangle(display_frame, (10, 10), (450, 45), (0, 0, 0), cv2.FILLED)
        cv2.putText(display_frame, f"LOCKING TARGETS: {progress}% ({len(current_list)} verified)", 
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        
        cv2.imshow(WINDOW_NAME, display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return []
            
        if stability_counter >= STABILITY_LIMIT:
            print(f"\n[SUCCESS] Stable tracking lock achieved!")
            return current_list

def phase_execute_batch(api, pick_list, drop_list):
    global current_object_idx, total_objects_count, machine_state
    time.sleep(0.5)
    
    if len(pick_list) == 0 or len(drop_list) == 0:
        print("[WARN] No targets or drop zones detected. Aborting.")
        return False
    
    machine_state = "pick place"
    total_objects_count = len(pick_list)
    print(f"\n[PHASE 3] Executing batch sequences for {total_objects_count} objects...")

    for i in range(total_objects_count):
        current_object_idx = i + 1  
        pick_x, pick_y, pick_r = pick_list[i]
        drop_x, drop_y = drop_list[0] 

        print("\n=======================================================")
        print(f" TARGETING OBJECT {current_object_idx}/{total_objects_count} -> APPROACHING ANGLE: {pick_r}°")
        print("=================================================")

        ensure_no_hand_or_pause(api)
        dobotArm.open_gripper(api)
        
        # Intercept frame rendering update right before movement commands
        safe_sleep_with_monitoring(api, 0.3) 
        
        print(f"[MOVE] Driving wrist servo to {pick_r}° and descending to pick coordinates...")
        move_safe_descend(api, pick_x, pick_y, Z_PICK_LOWER, pick_r)
        
        dobotArm.close_gripper(api)
        print("[INFO] Closing gripper... waiting for physical grab.")
        safe_sleep_with_monitoring(api, 1.5)  
        
        move_safe_ascend(api, pick_x, pick_y, pick_r)
        safe_sleep_with_monitoring(api, 0.2)   

        print("[SAFETY] Resetting wrist orientation axis to 0.0° for transit...")
        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE, rHead=0.0)
        safe_sleep_with_monitoring(api, 0.4)

        ensure_no_hand_or_pause(api)
        move_between_points(api, (pick_x, pick_y), (drop_x, drop_y), rHead=0.0)
        safe_sleep_with_monitoring(api, 0.8)   

        ensure_no_hand_or_pause(api)
        move_safe_descend(api, drop_x, drop_y, Z_PICK, rHead=0.0)
        
        dobotArm.open_gripper(api)
        dobotArm.stop_pump(api)
        print("[INFO] Releasing object... waiting for drop.")
        safe_sleep_with_monitoring(api, 1.5)  
        
        move_safe_ascend(api, drop_x, drop_y, rHead=0.0)
        safe_sleep_with_monitoring(api, 0.5)  
        print(f"[SUCCESS] Object {current_object_idx} deposited cleanly.")

    print("\n[PHASE 3] All detected objects cleared.")
    return True


# ---------------------------------------------------------
# MAIN EXECUTION PIPELINE
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n[SYSTEM] Connecting to Dobot hardware API...")
    dobotArm.initialize_robot(api)

    print("\n=======================================================")
    print("[SYSTEM] LAUNCHING AUTOMATION TRACK DIRECTLY")
    print("=======================================================")

    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)
    dType.SetQueuedCmdStartExec(api) 
    time.sleep(0.1)

    dobotArm.open_gripper(api)
    dobotArm.stop_pump(api)

    print("[SYSTEM STATUS] Launching background MediaPipe safety thread...")
    print("=======================================================\n")

    safety_thread = threading.Thread(target=background_safety_monitor_worker, args=(api,), daemon=True)
    safety_thread.start()

    try:
        print(f"[SYSTEM] Clearing workspace view. Moving to: X={CLEAR_X}, Y={CLEAR_Y}...")
        current_index = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, CLEAR_X, CLEAR_Y, CLEAR_Z, CLEAR_R, isQueued=0)[0]
        
        while True:
            queued_idx = dType.GetQueuedCmdCurrentIndex(api)[0]
            if queued_idx >= current_index:
                break
            time.sleep(0.05)

        while True:
            try:
                dType.SetQueuedCmdStopExec(api)
                dType.SetQueuedCmdClear(api)
                dType.SetQueuedCmdStartExec(api) 
                time.sleep(0.1)

                machine_state = "paused"
                
                print("\n=======================================================")
                print("[PRE-SCAN HOLD] Workspace clear window open.")
                print(" -> Arrange your setup, step away from the active arena,")
                print(" -> Then click on the video frame window and press SPACEBAR to begin scanning.")
                print("=======================================================")
                
                while True:
                    with frame_lock:
                        hold_frame = latest_processed_frame.copy()
                    
                    hold_frame = draw_visual_overlays(hold_frame)
                    
                    # Draw an overlay notice across the idle feed screen
                    cv2.rectangle(hold_frame, (10, 15), (630, 65), (139, 0, 0), cv2.FILLED)
                    cv2.putText(hold_frame, "SYSTEM INITIALIZED: SYSTEM PAUSED", (20, 38), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(hold_frame, "Press SPACEBAR inside this window to begin automation scan track", (20, 55), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    
                    cv2.imshow(WINDOW_NAME, hold_frame)
                    
                    key = cv2.waitKey(30) & 0xFF
                    if key == 32:   
                        print("[RUN] Spacebar signal caught. Beginning environment scan tracks...")
                        break
                    if key == ord('q'):
                        print("[SYSTEM] Exit token received.")
                        raise KeyboardInterrupt

                # --- START AUTOMATION PIPELINE ---
                drop_zone = phase_detect_plates()
                if drop_zone is None or len(drop_zone) == 0:
                    print("[ERROR] No drop zones detected. Restarting scan loop.")
                    continue

                pick_target = phase_detect_targets(drop_zone)
                if pick_target is None or len(pick_target) == 0:
                    print("[INFO] No valid targets left to process. Restarting loop.")
                    if not wait_for_space_to_restart():
                        break
                    continue

                completed = phase_execute_batch(api, pick_target, drop_zone)
                
                if completed:
                    print("[INFO] Batch complete. System idle, waiting for next instructions...")
                else:
                    print("[WARN] Batch execution did not complete successfully.")
                    continue

                try:
                    dType.SetQueuedCmdStopExec(api)
                    dType.SetQueuedCmdClear(api)
                except Exception:
                    pass
                
                if not wait_for_space_to_restart():
                    break

            except InterruptedError:
                print("\n[SYSTEM RESET] Interruption caught. Forcing physical arm safety parking move...")
                machine_state = "paused"
                try:
                    dType.SetQueuedCmdStartExec(api)
                    current_index = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, CLEAR_X, CLEAR_Y, CLEAR_Z, CLEAR_R, isQueued=0)[0]
                    while True:
                        if dType.GetQueuedCmdCurrentIndex(api)[0] >= current_index:
                            break
                        # Allow visual output responsiveness during redirection parking moves
                        with frame_lock:
                            f = latest_processed_frame.copy()
                        f = draw_visual_overlays(f)
                        cv2.imshow(WINDOW_NAME, f)
                        cv2.waitKey(20)
                except Exception as e:
                    print(f"[RECOVERY WARN] Safe clear command redirect failed: {e}")
                continue

    except KeyboardInterrupt:
        print("\n[SYSTEM] Manual termination sequence triggered.")

    finally:
        print("\n[SYSTEM shut down] Cleaning up resources...")
        try:
            dType.SetQueuedCmdStopExec(api)
            dType.SetQueuedCmdClear(api)
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()