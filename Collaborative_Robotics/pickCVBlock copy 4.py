import dobotArm
import lib.DobotDllType as dType
import numpy as np
import cv2
import time
import threading

# --- MEDIAPIPE SAFETY FRAMEWORK IMPORTS ---
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

"""CONSTANTS"""
Z_SAFE = 40          
Z_PICK = -15         
Z_PICK_LOWER = -35   
STABILITY_LIMIT = 60 
PIXEL_TOLERANCE = 10 
TARGET_MIN_AREA = 20  
TARGET_MAX_AREA = 4000 
HAND_CLEAR_FRAMES = 30 
HAND_DETECTION_ENABLED = True 

# --- CUSTOM RETRACTION COORDINATES ---
# Change these values to a physical location where the arm is completely out of the camera frame
CLEAR_X = 200.0
CLEAR_Y = -150.0
CLEAR_Z = 60.0
CLEAR_R = 0.0

# Shared Global Thread Communication Variables
machine_state = "scanning plate" 
HAND_IN_WORKSPACE = False
current_object_idx = 0
total_objects_count = 0

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
detector = vision.HandLandmarker.create_from_options(options)

# Load calibration targets
H_matrix = np.load("HomographyMatrix.npy")
data = np.load("./camera_params.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs   = data["dist_coeffs"]

# Compute undistort maps once using the correct 640x480 frame base
ret, frame = cap.read()
if frame is None:
    print("[CRITICAL ERROR] Could not read initial frame from camera source.")
    exit(1)
h, w = frame.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1)
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2)

# Create the persistent, single unified window
WINDOW_NAME = "Robot System Feed"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

# Global variables to pass processed video and frame data back to the main thread seamlessly
latest_processed_frame = frame.copy()
frame_lock = threading.Lock()


def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]


def move_safe_descend(api, x, y, z, rHead=0):
    dobotArm.move_to_xyz(api, x, y, Z_SAFE, rHead)
    dobotArm.move_to_xyz(api, x, y, z, rHead)


def move_safe_ascend(api, x, y, rHead=0):
    dobotArm.move_to_xyz(api, x, y, Z_SAFE, rHead)


def move_between_points(api, start, end, rHead=0):
    """
    Forces clean high-altitude transition directly from point A to point B at Z_SAFE level.
    """
    dobotArm.move_to_xyz(api, start[0], start[1], Z_SAFE, rHead)
    dobotArm.move_to_xyz(api, end[0], end[1], Z_SAFE, rHead)


# --- BACKGROUND SAFETY MONITOR WORKER THREAD ---
def background_safety_monitor_worker(api):
    global HAND_IN_WORKSPACE, current_object_idx, total_objects_count, latest_processed_frame
    
    cleared_frames = 0
    print("[SYSTEM] Background MediaPipe Safety & Video Thread Launched.")
    
    while cap.isOpened():
        ret, f = cap.read()
        if not ret or f is None:
            time.sleep(0.01)
            continue
            
        f = cv2.remap(f, map1, map2, cv2.INTER_LINEAR)
        display = f.copy()
        h_f, w_f = f.shape[:2] 
        
        with frame_lock:
            latest_processed_frame = f.copy()
        
        rgb_frame = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        image_rect = mp.tasks.components.containers.NormalizedRect(
            x_center=0.5, y_center=0.5, width=1.0, height=1.0, rotation=0.0
        )
        
        try:
            img_processing_options = vision.ImageProcessingOptions(region_of_interest=image_rect)
            detection_result = detector.detect(mp_image, img_processing_options)
        except AttributeError:
            detection_result = detector.detect(mp_image)
        
        if HAND_DETECTION_ENABLED and detection_result.hand_landmarks:
            HAND_IN_WORKSPACE = True
            cleared_frames = 0
            
            try:
                dType.SetQueuedCmdForceStopExec(api) 
                dType.SetQueuedCmdClear(api)
            except Exception:
                pass
                
            cv2.rectangle(display, (10, 20), (w_f - 10, 90), (0, 0, 255), cv2.FILLED)
            cv2.putText(display, "!! EMERGENCY STOP: HAND IN WORKSPACE !!", (25, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cleared_frames += 1
            if cleared_frames >= HAND_CLEAR_FRAMES:
                HAND_IN_WORKSPACE = False
                
            if machine_state == "pick place":
                cv2.rectangle(display, (10, 20), (420, 90), (0, 0, 0), cv2.FILLED)
                cv2.putText(display, f"RUNNING: Object {current_object_idx} of {total_objects_count}", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display, f"Status: Safe ({cleared_frames}/{HAND_CLEAR_FRAMES} frames clear)", 
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            elif machine_state == "scanning plate" or machine_state == "scanning target":
                continue
                
        cv2.imshow(WINDOW_NAME, display)
        cv2.waitKey(1)


def ensure_no_hand_or_pause(api):
    """Blocks execution on the robot loop if a safety condition is flagged by the worker thread."""
    global machine_state
    if HAND_IN_WORKSPACE:
        print("\n[SAFETY INTERCEPT] Threat signature detected! Executing instant drop sequence...")
        
        try:
            # 1. Kill the active motion queue instantly to freeze physical movement
            dType.SetQueuedCmdForceStopExec(api)
            dType.SetQueuedCmdClear(api)
            
            # 2. Re-engage immediate queue execution context to handle safety escapes
            dType.SetQueuedCmdStartExec(api)
            time.sleep(0.05)
            
            # 3. Drop the object immediately right where it froze
            print("[SAFETY] Opening gripper to release payload...")
            dobotArm.open_gripper(api)
            dobotArm.stop_pump(api)
            time.sleep(0.5) # Give the pneumatic valves a brief moment to vent pressure
            
            # 4. Move straight to the dedicated clear vantage point at a safe altitude
            print(f"[SAFETY] Retracting arm to clear view position: X={CLEAR_X}, Y={CLEAR_Y}...")
            dobotArm.move_to_xyz(api, CLEAR_X, CLEAR_Y, CLEAR_Z, CLEAR_R)
            
        except Exception as e:
            print(f"[SAFETY ERROR] Failed to execute physical escape sequence: {e}")
            
        # Block script execution here as long as the user's hand remains under the lens
        while HAND_IN_WORKSPACE:
            time.sleep(0.1)
            
        print("[SAFETY INTERCEPT] Hazard cleared. Workspace open. Re-initializing camera scan tracks.")
        
        # 5. Set the state engine back to plate scanning so it captures the new state of the workspace
        machine_state = "scanning plate"
        raise InterruptedError("Automation track safely broken due to workspace intrusion.")


def safe_sleep_with_monitoring(api, duration):
    """Sleep utility allowing safety flag evaluation to remain non-blocking."""
    start_time = time.time()
    while time.time() - start_time < duration:
        ensure_no_hand_or_pause(api)
        time.sleep(0.1)


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
        key = cv2.waitKey(100) & 0xFF
        if key == 32:  
            return True
        if key == ord('q'):
            return False


def phase_detect_plates():
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
                cv2.circle(display_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                rx, ry = pixel_to_robot(i[0], i[1], H_matrix)
                current_list.append((rx, ry))

        if len(current_list) > 0 and len(current_list) == last_count:
            stability_counter += 1
        else:
            stability_counter = 0
            last_count = len(current_list)

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        cv2.putText(display_frame, f"LOCKING PLATES: {progress}% ({len(current_list)} found)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow(WINDOW_NAME, display_frame)
        cv2.waitKey(1)

        if stability_counter >= STABILITY_LIMIT:
            print(f"Locked {len(current_list)} plates.")
            return current_list


def phase_detect_targets(drop_list):
    print("\n[PHASE 2] Scanning for targets. Waiting for stability...")
    EXCLUSION_RADIUS_MM = 20.0 
    stability_counter = 0
    last_count = -1
    
    while True:
        ensure_no_hand_or_pause(api)
        with frame_lock:
            frame = latest_processed_frame.copy()
            
        # Create a clean display copy and a clean drawing overlay frame
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
        
        # Temp storage to print stable outputs cleanly to console
        print_strings = []
        
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if TARGET_MIN_AREA < area < TARGET_MAX_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h) if h != 0 else 0
                fill_ratio = area / float(w * h) if w * h != 0 else 0
                
                if 0.15 < aspect_ratio < 6.0 and fill_ratio > 0.15:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        rx, ry = pixel_to_robot(cx, cy, H_matrix)
                        
                        is_inside_plate = False
                        for drop_x, drop_y in drop_list:
                            distance = np.sqrt((rx - drop_x)**2 + (ry - drop_y)**2)
                            if distance < EXCLUSION_RADIUS_MM:
                                is_inside_plate = True
                                break
                        
                        if is_inside_plate:
                            cv2.circle(overlay_frame, (cx, cy), 5, (0, 0, 255), -1)
                            continue 

                        # Continuous mathematical angle extraction
                        rect = cv2.minAreaRect(cnt)
                        (cx_box, cy_box), (box_w, box_h), angle = rect

                        if box_w < box_h:
                            absolute_angle = angle + 90
                        else:
                            absolute_angle = angle

                        absolute_angle = absolute_angle % 180
                        grasp_angle = (absolute_angle + 0) % 180 
                        pick_r = int(grasp_angle)
                        
                        current_list.append((rx, ry, pick_r))

                        # Draw the persistent outlines and angles onto the overlay layer
                        box = cv2.boxPoints(rect)
                        box = np.int64(box)
                        cv2.drawContours(overlay_frame, [box], 0, (255, 255, 0), 2) # Cyan bounding box
                        cv2.drawContours(overlay_frame, [cnt], 0, (0, 255, 0), 1)   # Green raw contour outline
                        cv2.putText(overlay_frame, f"Obj {len(current_list)}: {pick_r}deg", (cx + 10, cy - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        # Add tracking log data to the console output frame
                        print_strings.append(f"  -> Object {len(current_list)} detected at ({rx:.1f}, {ry:.1f}) | Target Angle: {pick_r}°")

        # Blend the persistent text/outlines overlay onto the active camera feed
        display_frame = cv2.addWeighted(display_frame, 1.0, overlay_frame, 1.0, 0)

        # Handle console output and stability verification updates
        if len(current_list) > 0 and len(current_list) == last_count:
            stability_counter += 1
            # Print angles to terminal only at milestone check-ins so it doesn't flood your console log
            if stability_counter % 15 == 0:
                print(f"\n[STABILITY CHECK] Frame count locked at {int((stability_counter/STABILITY_LIMIT)*100)}%:")
                for log in print_strings:
                    print(log)
        else:
            stability_counter = 0
            last_count = len(current_list)

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        cv2.rectangle(display_frame, (10, 10), (450, 45), (0, 0, 0), cv2.FILLED)
        cv2.putText(display_frame, f"LOCKING TARGETS: {progress}% ({len(current_list)} verified)", 
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        
        cv2.imshow(WINDOW_NAME, display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return []
            
        if stability_counter >= STABILITY_LIMIT:
            print(f"\n[SUCCESS] Stable tracking lock achieved!")
            print(f"=======================================================")
            for final_log in print_strings:
                print(final_log)
            print(f"=======================================================\n")
            return current_list

def phase_execute_batch(api, pick_list, drop_list):
    global current_object_idx, total_objects_count
    time.sleep(0.5)
    
    if len(pick_list) == 0 or len(drop_list) == 0:
        print("[WARN] No targets or drop zones detected. Aborting.")
        return False
    
    total_objects_count = len(pick_list)
    print(f"\n[PHASE 3] Executing batch sequences for {total_objects_count} objects...")

    for i in range(total_objects_count):
        current_object_idx = i + 1  
        pick_x, pick_y, pick_r = pick_list[i]
        drop_x, drop_y = drop_list[0] 

        # --- NEW: VISUAL TERMINAL LINE PRINTING EXPECTED DEGREES ---
        print("\n=======================================================")
        print(f" TARGETING OBJECT {current_object_idx}/{total_objects_count} -> APPROACHING ANGLE: {pick_r}°")
        print("=======================================================")

        # --- 1. PICK SEQUENCE ---
        ensure_no_hand_or_pause(api)
        dobotArm.open_gripper(api)
        time.sleep(0.3) 
        
        # Descend with targeted object rotation angle
        print(f"[MOVE] Driving wrist servo to {pick_r}° and descending to pick coordinates...")
        move_safe_descend(api, pick_x, pick_y, Z_PICK_LOWER, pick_r)
        
        dobotArm.close_gripper(api)
        print("[INFO] Closing gripper... waiting for physical grab.")
        safe_sleep_with_monitoring(api, 1.5)  
        
        # Ascend while keeping object orientation steady
        move_safe_ascend(api, pick_x, pick_y, pick_r)
        safe_sleep_with_monitoring(api, 0.2)   

        # --- SAFETY INJECTION: SNAP WRIST BACK TO 0.0 DEGREES ---
        print("[SAFETY] Resetting wrist orientation axis to 0.0° for transit...")
        dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE, rHead=0.0)
        safe_sleep_with_monitoring(api, 0.4)

        # --- 2. TRANSFER SEQUENCE ---
        ensure_no_hand_or_pause(api)
        # Transits directly to drop zone using safe 0-degree angle
        move_between_points(api, (pick_x, pick_y), (drop_x, drop_y), rHead=0.0)
        safe_sleep_with_monitoring(api, 0.8)   

        # --- 3. PLACE SEQUENCE ---
        ensure_no_hand_or_pause(api)
        # Descend flat into target pan
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
# MAIN EXECUTION PIPELINE (STRIPPED & ULTRA-FAST BOOT)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n[SYSTEM] Connecting to Dobot hardware API...")
    dobotArm.initialize_robot(api)

    print("\n=======================================================")
    print("[SYSTEM] LAUNCHING AUTOMATION TRACK DIRECTLY")
    print("=======================================================")

    # 1. Clear out any lingering command memories from previous runs
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)
    dType.SetQueuedCmdStartExec(api) 
    time.sleep(0.1)

    # 2. Set peripheral end-effectors to known clean defaults immediately
    dobotArm.open_gripper(api)
    dobotArm.stop_pump(api)

    print("[SYSTEM STATUS] Launching background MediaPipe safety thread...")
    print("=======================================================\n")

    # Spin up the asynchronous video and safety thread once
    safety_thread = threading.Thread(target=background_safety_monitor_worker, args=(api,), daemon=True)
    safety_thread.start()

    try:
        while True:
            try:
                # Reset queues per main automation cycle loop
                dType.SetQueuedCmdStopExec(api)
                dType.SetQueuedCmdClear(api)
                dType.SetQueuedCmdStartExec(api) 
                time.sleep(0.1)

                machine_state = "scanning plate"
                print("\n[RUN] Starting automation scan cycle...")

                # --- PHASE 1: DROP ZONES ---
                drop_zone = phase_detect_plates()
                if drop_zone is None or len(drop_zone) == 0:
                    print("[ERROR] No drop zones detected. Restarting scan loop.")
                    continue
                next_state()

                # --- PHASE 2: TARGET OBJECTS ---
                pick_target = phase_detect_targets(drop_zone)
                if pick_target is None or len(pick_target) == 0:
                    print("[INFO] No valid targets left to process. Restarting loop.")
                    if not wait_for_space_to_restart():
                        break
                    continue
                next_state()

                # --- PHASE 3: BATCH PROCESSING RUN ---
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
                print("[SYSTEM RESET] Re-initializing state tracking variables for safety...")
                continue

    finally:
        print("\n[SYSTEM shut down] Cleaning up resources...")
        try:
            dType.SetQueuedCmdStopExec(api)
            dType.SetQueuedCmdClear(api)
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()