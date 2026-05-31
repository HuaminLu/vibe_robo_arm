#This code is a simplified implementation of a collaborative robotics system that detects plates and targets using computer vision, 
#and then commands a Dobot robotic arm to pick and place objects accordingly. The system operates in three phases: scanning for plates, 
#scanning for targets, and executing the pick/place operations. 
#Stability checks are implemented to ensure reliable detection before proceeding to the next phase.

# Note: there are parameters that are useful to the successful operation of the robot arm. Read through the code before running the program.

# How to use: 
# 1. Ensure you have the Dobot robotic arm set up and connected to your computer.
# 2. Place the plates (drop zones) and targets (red blocks) within the camera's
# field of view.
# 3. Run the script. The system will first scan for plates, then targets, and finally execute the pick/place operations based on the detected positions.
# 4. Monitor the console output and the video feed for feedback on the system's status and operations

#Other Useful Codes you can use:
#dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE, rHead): moves the robot to the specified (x, y, z) coordinates with a specified rotation for the end effector (rHead). Z_SAFE is a predefined constant that ensures the robot maintains a safe height to avoid collisions when moving horizontally.



import dobotArm
import lib.DobotDllType as dType
import numpy as np
import cv2
import time


"""CONSTANTS"""

Z_SAFE = 40 #what is the clearance distance for the robot arm to avoid collisions when moving horizontally?
Z_PICK = -15 # normal drop height for placing objects
Z_PICK_LOWER = -35 # lower pick height so the gripper reaches the object reliably
STABILITY_LIMIT = 60  #how many consecutive frames of stable detection before we "lock in" the positions and move to the next phase? (at 30fps, 60 frames is about 2 seconds)
PIXEL_TOLERANCE = 10  #object can move at most this # of pixels to be considered stationary
TARGET_MIN_AREA = 100  # minimum area for a velcro patch in pixels (lowered for small velcro)
TARGET_MAX_AREA = 4000  # maximum area for a velcro patch to avoid picking up a whole part
HAND_CLEAR_FRAMES = 30   # number of consecutive frames with no hand before resuming
HAND_DETECTION_ENABLED = False  # disable hand detection while tuning

machine_state = "scanning plate" 

# --- INITIALIZATION FOR CAMERA TRANSFORMATION ---
# MAKE SURE THAT YOU HAVE RAN calibrateCamera.py FIRST TO GENERATE THE camera_params.npz FILE
api = dType.load()
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("[WARN] Camera index 0 failed, trying index 1...")
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("[ERROR] Unable to open camera. Check the camera index and connection.")
    exit(1)
H_matrix = np.load("HomographyMatrix.npy")
data = np.load("./camera_params.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs   = data["dist_coeffs"]

# Compute undistort maps once
ret, frame = cap.read()
h, w = frame.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1)
map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w,h), cv2.CV_16SC2)

def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]


def move_safe_descend(api, x, y, z, rHead=0):
    """Approach from above, then descend cleanly to the pick/place height."""
    dobotArm.move_to_xyz(api, x, y, Z_SAFE, rHead)
    dobotArm.move_to_xyz(api, x, y, z, rHead)


def move_safe_ascend(api, x, y, rHead=0):
    """Return straight up to a safe clearance height after pick/place."""
    dobotArm.move_to_xyz(api, x, y, Z_SAFE, rHead)


def move_between_points(api, start, end, rHead=0):
    """Use a midway safe waypoint to smooth motion between two XY points."""
    mid_x = (start[0] + end[0]) / 2.0
    mid_y = (start[1] + end[1]) / 2.0
    dobotArm.move_to_xyz(api, mid_x, mid_y, Z_SAFE, rHead)
    dobotArm.move_to_xyz(api, end[0], end[1], Z_SAFE, rHead)


def detect_hand(frame):
    """Detect a human hand using a combined skin-color mask and shape filtering."""
    if not HAND_DETECTION_ENABLED:
        return False, np.zeros(frame.shape[:2], dtype=np.uint8), None

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 135, 85]), np.array([255, 180, 135]))
    mask_hsv = cv2.inRange(hsv, np.array([0, 10, 60]), np.array([25, 150, 255]))
    hand_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

    hand_mask = cv2.GaussianBlur(hand_mask, (7, 7), 0)
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame.shape[0] * frame.shape[1]
    min_area = 5000
    max_area = int(frame_area * 0.12)

    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h != 0 else 1.0
        if aspect_ratio < 0.35 or aspect_ratio > 2.8:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        if solidity < 0.35:
            continue

        best_cnt = cnt
        break

    if best_cnt is not None:
        return True, hand_mask, best_cnt
    return False, hand_mask, None


def ensure_no_hand_or_pause(api):
    if not HAND_DETECTION_ENABLED:
        return

    """If a hand is seen, retreat the robot to a safe home and pause until the hand clears.
    This blocks until the frame has been hand-free for HAND_CLEAR_FRAMES consecutive frames.
    """
    """If a hand is seen, retreat the robot to a safe home and pause until the hand clears.
    This blocks until the frame has been hand-free for HAND_CLEAR_FRAMES consecutive frames.
    """
    cleared = 0
    while True:
        ret, f = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        hand, mask, cnt = detect_hand(f)
        display = f.copy()
        if hand:
            # Immediate safety action: stop queued commands (best-effort) and retreat to home
            try:
                dType.SetQueuedCmdStopExec(api)
                dType.SetQueuedCmdClear(api)
            except Exception:
                pass
            try:
                dobotArm.move_to_home(api)
            except Exception:
                pass

            cv2.putText(display, "HAND DETECTED - PAUSED", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if cnt is not None:
                cv2.drawContours(display, [cnt], -1, (0, 0, 255), 2)
            cv2.imshow("Detection", display)
            cv2.waitKey(1)
            cleared = 0
            print("[SAFETY] Human hand detected — robot paused and retreated to home.")
            # keep looping until cleared for HAND_CLEAR_FRAMES frames
            time.sleep(0.1)
            continue
        else:
            cv2.putText(display, "No hand detected", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow("Detection", display)
            cv2.waitKey(1)
            cleared += 1
            if cleared >= HAND_CLEAR_FRAMES:
                print("[SAFETY] Hand cleared. Resuming operations.")
                return


# State machine logic to control the flow of the program through the three phases: scanning for plates, scanning for targets, and executing pick/place operations.
# THIS STATE MACHINE IS TOO SIMPLE. Can you think of logics that should change the robot's sequnece of actions?
# Ex: what if the robot fails to pick up a target? should it retry? should it go back to scanning for targets in case the target was moved? what if a new plate is added during the pick/place phase?
# What if a human's hand is in sight during pick/place phase? (safety first!)

def next_state():
    global machine_state
    if machine_state == "scanning plate":
        machine_state = "scanning target"
    elif machine_state == "scanning target":
        machine_state = "pick place"
    elif machine_state == "pick place":
        machine_state = "scanning plate"
    else:
        machine_state = "scanning plate"


def wait_for_space_to_restart():
    print("[INFO] Run complete. Press SPACE to scan again, or Q to quit.")
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == 32:  # SPACE
            return True
        if key == ord('q'):
            return False



def phase_detect_plates():
    print("\n[PHASE 1] Scanning for drop zones. Waiting for stability...")
    stability_counter = 0
    last_count = 0
    
    while True:
        ret, frame = cap.read()
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=35, minRadius=25, maxRadius=55)

        current_list = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(display_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                rx, ry = pixel_to_robot(i[0], i[1], H_matrix)
                current_list.append((rx, ry))

        # --- AUTO-LOCK LOGIC ---
        if len(current_list) > 0 and len(current_list) == last_count:
            stability_counter += 1
        else:
            stability_counter = 0
            last_count = len(current_list)

        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        cv2.putText(display_frame, f"LOCKING PLATES: {progress}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Detection", display_frame)
        cv2.waitKey(1)

        if stability_counter >= STABILITY_LIMIT:
            print(f"Locked {len(current_list)} plates.")
            return current_list
  
# ---------------------------------------------------------
# PHASE 2: DETECT Red velcros to pick up (Red Blocks)
# this script assumes the targets to be picked up are red blocks
# be aware your target maynot be red, and they may not be rectangular! You will need to modify the detection logic to fit your specific use case.
# ---------------------------------------------------------
def phase_detect_targets():
    print("\n[PHASE 2] Scanning for targets. Waiting for stability...")
    stability_counter = 0
    last_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        # Create a display copy so drawings don't affect next frame's HSV detection
        display_frame = frame.copy()
        if HAND_DETECTION_ENABLED:
            hand_detected, hand_mask, hand_cnt = detect_hand(frame)
            if hand_detected:
                cv2.putText(display_frame, "HAND DETECTED! PAUSE", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.drawContours(display_frame, [hand_cnt], -1, (0, 0, 255), 2)
                print("[ALERT] Human hand detected in camera frame!")
                stability_counter = 0
                last_count = -1
                cv2.imshow("Detection", display_frame)
                cv2.waitKey(1)
                continue
        
        # Red Tag Logic (improved with stricter color and circularity)
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5,5), 0), cv2.COLOR_BGR2HSV)
        # Stricter HSV: require higher saturation (avoid whites and grays)
        lower_red1 = np.array([0, 120, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 120, 100])
        upper_red2 = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        hsv_red = cv2.bitwise_or(mask1, mask2)

        # BGR dominance filter to help reject brown wood and shadows (use safe int types)
        red_channel = frame[:, :, 2].astype(np.int16)
        green_channel = frame[:, :, 1].astype(np.int16)
        blue_channel = frame[:, :, 0].astype(np.int16)
        red_dom = (red_channel > 140) & (red_channel > green_channel + 60) & (red_channel > blue_channel + 60)
        red_dom_mask = (red_dom.astype(np.uint8) * 255)

        # Combine HSV and dominance masks; fall back to HSV-only if dominance removes everything
        combined = cv2.bitwise_and(hsv_red, red_dom_mask)
        if cv2.countNonZero(combined) == 0 and cv2.countNonZero(hsv_red) > 0:
            mask = hsv_red  # looser fallback
            debug_note = 'loose-hsv'
        else:
            mask = combined
            debug_note = 'combined'

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        # Debug windows to help tune thresholds (small, non-blocking)
        cv2.imshow("HSV Red Mask", hsv_red)
        cv2.imshow("Red Dominance Mask", red_dom_mask)
        cv2.imshow("Final Mask", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Debug logging
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            print(f"[DETECT] {len(contours)} contours (max area {max(areas):.0f}) using {debug_note}")

        current_list = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if TARGET_MIN_AREA < area < TARGET_MAX_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h) if h != 0 else 0
                fill_ratio = area / float(w * h) if w * h != 0 else 0
                if 0.25 < aspect_ratio < 4.0 and fill_ratio > 0.25:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        rx, ry = pixel_to_robot(cx, cy, H_matrix)
                        rect = cv2.minAreaRect(cnt)
                        angle = rect[2]
                        if w < h:
                            angle += 90
                        angle = int((angle + 180) % 180 - 90)
                        current_list.append((rx, ry, angle))
                        cv2.drawContours(display_frame, [cnt], -1, (0, 255, 0), 2)
                        box = cv2.boxPoints(rect)
                        box = np.int32(box)
                        cv2.drawContours(display_frame, [box], 0, (255, 0, 0), 2)
                        cv2.putText(display_frame, f"{angle}deg", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
        cv2.waitKey(1)

        # --- STABILITY LOGIC ---
        if len(current_list) != 0:
            if len(current_list) > 0 and len(current_list) == last_count:
                stability_counter += 1
            else:
                stability_counter = 0
                last_count = len(current_list)

        # Visual Feedback
        progress = int((stability_counter / STABILITY_LIMIT) * 100)
        color = (0, 255, 0) if progress < 100 else (255, 255, 0)
        
        cv2.putText(display_frame, f"LOCKING TARGETS: {progress}%", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Detection", display_frame)
        
        # --- EXIT CONDITION ---
        if stability_counter >= STABILITY_LIMIT:
            print(f"[SUCCESS] Locked {len(current_list)} targets.")
            #cv2.waitKey(500) # Brief pause so you can see the 100%
    
            return current_list


# ---------------------------------------------------------
# PHASE 3: PICK/PLACE LOOP
# This function assumes 1 drop zone only has 1 part, and executes the pick/place operations in batches.
# if you are picking up rigid car parts, would you still be able to move directly to the object and to the drop zone? 
# Do you need collision avoidance? Think about if the robot gripper accidentally hits the plate or other parts on the way to the target, what would happen? How would you modify the robot's movement logic to avoid collisions?
# ---------------------------------------------------------
def phase_execute_batch(api, pick_list, drop_list):
    time.sleep(0.5)
    
    if len(pick_list) == 0 or len(drop_list) == 0:
        print("missing targets, aborting")
        return False
    
    # If there are more targets than plates, drop all targets at the first plate.
    batch_size = min(len(pick_list), len(drop_list))
    print(f"\n[PHASE 3] Executing batch of {batch_size} operations.")

    for i in range(batch_size):
        pick_x, pick_y, pick_r = pick_list[i]
        drop_x, drop_y = drop_list[i] if len(drop_list) > 1 else drop_list[0]

        # Before starting any motion, ensure no human hand is in the workspace
        ensure_no_hand_or_pause(api)

        print(f"Task {i+1}: Moving {pick_x, pick_y} to {drop_x, drop_y} with orientation {pick_r}")

        # --- PICK SEQUENCE ---
        dobotArm.open_gripper(api)
        ensure_no_hand_or_pause(api)
        move_safe_descend(api, pick_x, pick_y, Z_PICK_LOWER, pick_r)
        dobotArm.close_gripper(api)
        move_safe_ascend(api, pick_x, pick_y, pick_r)

        # --- TRANSFER SEQUENCE ---
        ensure_no_hand_or_pause(api)
        move_between_points(api, (pick_x, pick_y), (drop_x, drop_y), pick_r)

        # --- PLACE SEQUENCE ---
        ensure_no_hand_or_pause(api)
        move_safe_descend(api, drop_x, drop_y, Z_PICK, pick_r)
        dobotArm.open_gripper(api)
        dobotArm.stop_pump(api)
        move_safe_ascend(api, drop_x, drop_y, pick_r)

    # irl, it is ok for 1 dish to contain multiple parts
    # if len(pick_list) > len(drop_list):
    #     for i in range(len(pick_list)):
    #         pick_x, pick_y = pick_list[i]
    #         drop_x, drop_y = drop_list[0]
    #         # --- PICK SEQUENCE ---
    #         dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)
    #         dobotArm.move_to_xyz(api, pick_x, pick_y, Z_PICK)
    #         dobotArm.close_gripper(api)
    #         dobotArm.move_to_xyz(api, pick_x, pick_y, Z_SAFE)

    #     # --- PLACE SEQUENCE ---
    #         dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)
    #         dobotArm.open_gripper(api)
    #         dobotArm.stop_pump(api)
    #         dobotArm.move_to_xyz(api, drop_x, drop_y, Z_SAFE)

    # Retreat to the side to clear the workspace for the next scan
    print("\n[PHASE 3] Retreating to side...")
    dobotArm.move_to_home(api)
    
    print("\nBatch Complete.")
    return True
 

# ---------------------------------------------------------
# FIXED MAIN EXECUTION
# ---------------------------------------------------------
dobotArm.initialize_robot(api)
dobotArm.open_gripper(api)
dobotArm.stop_pump(api)

# 1. Clear AND explicitly start the command queue engine right at the beginning
dType.SetQueuedCmdClear(api)
dType.SetQueuedCmdStartExec(api) 

try:
    while True:
        machine_state = "scanning plate"
        print("\n[RUN] Starting scan cycle...")

        drop_zone = phase_detect_plates()
        if drop_zone is None:
            print("[ERROR] No drop zones detected. Restarting scan loop.")
            continue
        next_state()

        pick_target = phase_detect_targets()
        if pick_target is None:
            print("[ERROR] No targets detected. Restarting scan loop.")
            continue
        next_state()

        # 2. Before executing commands, make sure the queue is running cleanly
        dType.SetQueuedCmdStartExec(api)

        completed = phase_execute_batch(api, pick_target, drop_zone)
        if completed:
            print("[INFO] Cycle complete. Press SPACE to run again or Q to quit.")
        else:
            print("[WARN] Batch execution did not complete successfully.")

        # 3. FIX: Wait for the physical robot to actually finish moving to its home 
        # position before clearing the queue! Otherwise, you cut its commands off mid-air.
        print("[INFO] Syncing with hardware... waiting for robot to reach home.")
        while True:
            # Check the index of the currently executing command
            current_idx = dType.GetQueuedCmdCurrentIndex(api)[0]
            # Check the index of the last command sent to the queue
            last_idx = dType.GetQueuedCmdCmdIndex(api)[0]
            if current_idx >= last_idx:
                break
            time.sleep(0.1)

        # 4. Now that the arm is completely stopped at home, safely clear the queue
        dType.SetQueuedCmdStopExec(api)
        dType.SetQueuedCmdClear(api)
        
        # 5. Prompt for the spacebar restart
        if not wait_for_space_to_restart():
            break

finally:
    # Safe shutdown cleanup
    dType.SetQueuedCmdStopExec(api)
    dType.SetQueuedCmdClear(api)
    cap.release()
    cv2.destroyAllWindows()