import dobotArm
import lib.DobotDllType as dType
import numpy as np
import cv2
import time
import importlib.util
import os

# --- DYNAMICALLY IMPORT YOUR AUTOMATIC FILE ---
# This safely loads "pickCVBlock copy 2.py" without renaming the file
auto_filename = "pickCVBlock copy 2.py"
if not os.path.exists(auto_filename):
    print(f"[ERROR] Could not find your automatic file: '{auto_filename}'")
    print("Make sure this controller is saved in the exact same directory!")
    exit(1)

spec = importlib.util.spec_from_file_location("auto_module", auto_filename)
auto_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto_module)
print(f"[SYSTEM] Successfully linked and loaded automatic logic from {auto_filename}")

# --- SYNCHRONIZE CONSTANTS FROM YOUR FILE ---
# We pull constants directly from your file so configurations match perfectly
Z_SAFE = getattr(auto_module, 'Z_SAFE', 40)
Z_PICK = getattr(auto_module, 'Z_PICK', -15)
Z_PICK_LOWER = getattr(auto_module, 'Z_PICK_LOWER', -35)
STABILITY_LIMIT = getattr(auto_module, 'STABILITY_LIMIT', 60)
TARGET_MIN_AREA = getattr(auto_module, 'TARGET_MIN_AREA', 50)
TARGET_MAX_AREA = getattr(auto_module, 'TARGET_MAX_AREA', 4000)

# Modes: "scanning plate", "scanning target", "pick place", or "MANUAL"
machine_state = "scanning plate"
gripper_state = "OPEN"
manual_target_str = "None"

# --- ACCESS SHARED COMPONENT INSTANCES ---
# Use the camera and Dobot API instances already setup in your file
api = auto_module.api
cap = auto_module.cap
H_matrix = auto_module.H_matrix
map1 = auto_module.map1
map2 = auto_module.map2

# --- MOUSE CALLBACK FOR MANUAL MODE ---
def click_to_move_callback(event, x, y, flags, param):
    global manual_target_str
    
    # ONLY respond to clicks if we are explicitly in MANUAL mode
    if machine_state == "MANUAL" and event == cv2.EVENT_LBUTTONDOWN:
        rx, ry = auto_module.pixel_to_robot(x, y, H_matrix)
        manual_target_str = f"X={rx:.1f}, Y={ry:.1f}"
        print(f"\n[MANUAL CLICK] Moving to target location: X={rx:.2f}mm, Y={ry:.2f}mm")
        
        dType.SetQueuedCmdClear(api)
        # Go over target safely at your file's Z_SAFE, then descend to Z_PICK
        dobotArm.move_to_xyz(api, rx, ry, Z_SAFE, 0)
        dobotArm.move_to_xyz(api, rx, ry, Z_PICK, 0)
        dType.SetQueuedCmdStartExec(api)

# --- HUD DRAWING HELPER ---
def draw_hud(img):
    # Background Box for clear visibility
    cv2.rectangle(img, (5, 5), (430, 110), (0, 0, 0), -1)
    
    # Orange text for manual intervention, green for automated processes
    mode_color = (0, 165, 255) if machine_state == "MANUAL" else (0, 255, 0)
    
    cv2.putText(img, f"MODE: {machine_state.upper()}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
    cv2.putText(img, f"Gripper: {gripper_state}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"Manual Coordinate: {manual_target_str}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "[M] Manual  [A] Auto  [O] Open  [C] Close  [S] Stop  [Q] Quit", (15, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

# --- KEYBOARD INTERRUPT CHECK ---
def check_keyboard_inputs():
    global machine_state, gripper_state
    
    key = cv2.waitKey(1) & 0xFF
    if key == 255: 
        return False  # No key hit
        
    if key == ord('q') or key == ord('Q'):
        return True # Signal shutdown
        
    elif key == ord('m') or key == ord('M'):
        if machine_state != "MANUAL":
            print("\n[MODE SWITCH] Switching to MANUAL control mode. Disabling automation tracks...")
            machine_state = "MANUAL"
            try:
                dType.SetQueuedCmdStopExec(api)
                dType.SetQueuedCmdClear(api)
                dType.SetQueuedCmdStartExec(api)
            except Exception: pass
            
    elif key == ord('a') or key == ord('A'):
        if machine_state == "MANUAL":
            print("\n[MODE SWITCH] Re-engaging AUTOMATIC routine tracks.")
            machine_state = "scanning plate"
            
    elif key == ord('o') or key == ord('O'):
        print("[MANUAL OVERRIDE] Opening Gripper...")
        dType.SetQueuedCmdClear(api)
        dobotArm.open_gripper(api)
        dType.SetQueuedCmdStartExec(api)
        gripper_state = "OPEN"
        
    elif key == ord('c') or key == ord('C'):
        print("[MANUAL OVERRIDE] Closing Gripper...")
        dType.SetQueuedCmdClear(api)
        dobotArm.close_gripper(api)
        dType.SetQueuedCmdStartExec(api)
        gripper_state = "CLOSED"
        
    elif key == ord('s') or key == ord('S'):
        print("[MANUAL OVERRIDE] Cutting pump air flow...")
        dType.SetQueuedCmdClear(api)
        dobotArm.stop_pump(api)
        dType.SetQueuedCmdStartExec(api)
        gripper_state = "PUMP STOPPED"
        
    return False

# Bind click window
cv2.namedWindow("Detection")
cv2.setMouseCallback("Detection", click_to_move_callback)

# Track parameters across frames
drop_zone = []
pick_target = []
stability_counter = 0
last_count = 0

print("\n" + "="*60)
print(" DUAL-MODE CONTROLLER BOOT SEQUENCE COMPLETE")
print("="*60)
print(" -> Running automatic loops from 'pickCVBlock copy 2.py'")
print(" -> Tap 'M' key at any time to freeze loop and take manual click control.")
print(" -> Tap 'A' key to hand control back to the computer vision script.")
print("="*60)

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None: 
            continue
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()
        
        # Continuous non-blocking check for manual keystroke changes
        if check_keyboard_inputs():
            break
            
        # --- MODE EXECUTION ROUTINES ---
        if machine_state == "MANUAL":
            # Idle video window loop showing your HUD waiting for mouse actions
            draw_hud(display_frame)
            cv2.imshow("Detection", display_frame)
            
        elif machine_state == "scanning plate":
            # Run the Plate detection algorithms from your original file
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.medianBlur(gray, 7)
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=35, minRadius=25, maxRadius=55)

            current_list = []
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(display_frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    rx, ry = auto_module.pixel_to_robot(i[0], i[1], H_matrix)
                    current_list.append((rx, ry))

            if len(current_list) > 0 and len(current_list) == last_count:
                stability_counter += 1
            else:
                stability_counter = 0
                last_count = len(current_list)

            progress = int((stability_counter / STABILITY_LIMIT) * 100)
            cv2.putText(display_frame, f"LOCKING PLATES: {progress}%", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            draw_hud(display_frame)
            cv2.imshow("Detection", display_frame)
            
            if stability_counter >= STABILITY_LIMIT:
                drop_zone = current_list
                print(f"[AUTO LOGIC] Locked {len(drop_zone)} mechanical plates.")
                stability_counter = 0
                machine_state = "scanning target"

        elif machine_state == "scanning target":
            # Run the spatial isolation logic from your file
            EXCLUSION_RADIUS_MM = 35.0 
            hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5,5), 0), cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 120, 100]); upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 120, 100]); upper_red2 = np.array([179, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1); mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            hsv_red = cv2.bitwise_or(mask1, mask2)
            
            contours, _ = cv2.findContours(hsv_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            current_targets = []
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if TARGET_MIN_AREA < area < TARGET_MAX_AREA:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        rx, ry = auto_module.pixel_to_robot(cx, cy, H_matrix)
                        
                        is_inside_plate = False
                        for drop_x, drop_y in drop_zone:
                            distance = np.sqrt((rx - drop_x)**2 + (ry - drop_y)**2)
                            if distance < EXCLUSION_RADIUS_MM:
                                is_inside_plate = True
                                break
                        
                        if is_inside_plate:
                            cv2.circle(display_frame, (cx, cy), 5, (0, 0, 255), -1)
                            continue
                            
                        rect = cv2.minAreaRect(cnt)
                        angle = rect[2]
                        current_targets.append((rx, ry, int(angle)))
                        cv2.drawContours(display_frame, [cnt], -1, (0, 255, 0), 2)
            
            draw_hud(display_frame)
            cv2.imshow("Detection", display_frame)
            
            if len(current_targets) > 0:
                pick_target = current_targets
                print(f"[AUTO LOGIC] Target detected. Handing over command ring buffer...")
                machine_state = "pick place"
            elif len(contours) == 0:
                machine_state = "scanning plate"

        elif machine_state == "pick place":
            draw_hud(display_frame)
            cv2.putText(display_frame, "EXECUTING BATCH TRANSLATIONS...", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Detection", display_frame)
            cv2.waitKey(1)
            
            # Directly executes the exact structural safety path variables matching your code
            if len(pick_target) > 0 and len(drop_zone) > 0:
                for i in range(len(pick_target)):
                    # Check keyboard input *between* items in a batch so you don't get locked out
                    if check_keyboard_inputs() or machine_state == "MANUAL": 
                        break
                        
                    pick_x, pick_y, pick_r = pick_target[i]
                    drop_x, drop_y = drop_zone[0]
                    
                    dobotArm.open_gripper(api)
                    time.sleep(0.3)
                    auto_module.move_safe_descend(api, pick_x, pick_y, Z_PICK_LOWER, pick_r)
                    
                    dobotArm.close_gripper(api)
                    time.sleep(1.5)
                    
                    auto_module.move_safe_ascend(api, pick_x, pick_y, pick_r)
                    auto_module.move_between_points(api, (pick_x, pick_y), (drop_x, drop_y), pick_r)
                    
                    auto_module.move_safe_descend(api, drop_x, drop_y, Z_PICK, pick_r)
                    dobotArm.open_gripper(api)
                    dobotArm.stop_pump(api)
                    time.sleep(1.5)
                    
                    auto_module.move_safe_ascend(api, drop_x, drop_y, pick_r)
                
                print("[AUTO LOGIC] Batch run complete. Homing mechanism.")
                dobotArm.move_to_home(api)
                time.sleep(1.5)
            
            machine_state = "scanning plate"

finally:
    print("[CLEANUP] Stopping hardware queues and closing camera feeds...")
    try:
        dType.SetQueuedCmdStopExec(api)
        dType.SetQueuedCmdClear(api)
    except Exception: pass
    cap.release()
    cv2.destroyAllWindows()