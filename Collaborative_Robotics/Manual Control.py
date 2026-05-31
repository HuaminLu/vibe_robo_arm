import dobotArm
import lib.DobotDllType as dType
import numpy as np
import cv2
import time

# --- CONSTANTS ---
Z_SAFE = 40       # Safe travel height
Z_MANUAL_SURFACE = -30  # Height the arm descends to after clicking a point

# --- INITIALIZATION ---
api = dType.load()
dobotArm.initialize_robot(api)

# Explicitly clean and start the Dobot command queue
dType.SetQueuedCmdClear(api)
dType.SetQueuedCmdStartExec(api)

# Load Camera and Transformation Parameters
try:
    H_matrix = np.load("HomographyMatrix.npy")
    data = np.load("./camera_params.npz")
    camera_matrix = data["camera_matrix"]
    dist_coeffs   = data["dist_coeffs"]
    print("[INFO] Successfully loaded camera parameters and homography matrix.")
except Exception as e:
    print(f"[ERROR] Failed to load calibration files: {e}")
    print("Please ensure HomographyMatrix.npy and camera_params.npz exist.")
    exit(1)

# Initialize Camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("[WARN] Camera index 1 failed, trying index 0...")
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Unable to open camera.")
    exit(1)

# Pre-compute undistort maps
ret, frame = cap.read()
if ret:
    h, w = frame.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1)
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_K, (w, h), cv2.CV_16SC2)
else:
    print("[ERROR] Could not read initial frame.")
    exit(1)

# --- GLOBAL TRACKING VARIABLES ---
target_robot_x = None
target_robot_y = None
gripper_state = "OPEN"

# --- HELPER FUNCTIONS ---
def pixel_to_robot(u, v, H):
    p = np.array([u, v, 1])
    xy = H @ p
    xy /= xy[2]
    return xy[0], xy[1]

# --- MOUSE CALLBACK FUNCTION ---
def click_to_move_callback(event, x, y, flags, param):
    global target_robot_x, target_robot_y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert the pixel (x, y) you clicked to physical robot coordinates
        rx, ry = pixel_to_robot(x, y, H_matrix)
        target_robot_x, target_robot_y = rx, ry
        
        print(f"\n[CLICK] Clicked Pixel: ({x}, {y}) -> Robot Space: (X: {rx:.2f}mm, Y: {ry:.2f}mm)")
        print(f"[MOVE] Moving to coordinates over safe clearance...")
        
        # 1. Clear any old commands just in case
        dType.SetQueuedCmdClear(api)
        
        # 2. Hover over the clicked position safely
        dobotArm.move_to_xyz(api, rx, ry, Z_SAFE, 0)
        
        # 3. Descend to the operating surface height
        dobotArm.move_to_xyz(api, rx, ry, Z_MANUAL_SURFACE, 0)
        
        # Ensure queue runs
        dType.SetQueuedCmdStartExec(api)

# Set up the OpenCV window and attach the click mouse listener
cv2.namedWindow("Dobot Manual Control")
cv2.setMouseCallback("Dobot Manual Control", click_to_move_callback)

print("\n" + "="*50)
print(" DOBOT MANUAL CLICK-TO-MOVE INTERFACE READY")
print("="*50)
print(" -> LEFT-CLICK anywhere on the frame to move the robot arm.")
print(" -> Press 'O' to OPEN the gripper.")
print(" -> Press 'C' to CLOSE the gripper.")
print(" -> Press 'S' to STOP the pump entirely.")
print(" -> Press 'Q' to Quit safely.")
print("="*50)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Fix lens distortion
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_frame = frame.copy()
        
        # --- HUD overlay instructions ---
        cv2.putText(display_frame, f"Gripper Status: {gripper_state}", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_frame, "[O] Open  [C] Close  [S] Pump Stop  [Q] Quit", (15, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw target crosshair if a spot was selected
        if target_robot_x is not None:
            cv2.putText(display_frame, f"Target: X={target_robot_x:.1f}, Y={target_robot_y:.1f}", 
                        (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Dobot Manual Control", display_frame)
        
        # Capture keyboard triggers (waiting 30ms per frame)
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\n[SHUTDOWN] Exiting control interface.")
            break
            
        elif key == ord('o') or key == ord('O'):
            print("[COMMAND] Opening Gripper...")
            dType.SetQueuedCmdClear(api)
            dobotArm.open_gripper(api)
            dType.SetQueuedCmdStartExec(api)
            gripper_state = "OPEN"
            
        elif key == ord('c') or key == ord('C'):
            print("[COMMAND] Closing Gripper...")
            dType.SetQueuedCmdClear(api)
            dobotArm.close_gripper(api)
            dType.SetQueuedCmdStartExec(api)
            gripper_state = "CLOSED"
            
        elif key == ord('s') or key == ord('S'):
            print("[COMMAND] Stopping Pump...")
            dType.SetQueuedCmdClear(api)
            dobotArm.stop_pump(api)
            dType.SetQueuedCmdStartExec(api)
            gripper_state = "PUMP STOPPED"

finally:
    # Safe shutdown cleanup
    print("[CLEANUP] Cleaning command ring buffer and shutting camera off...")
    try:
        dType.SetQueuedCmdStopExec(api)
        dType.SetQueuedCmdClear(api)
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()