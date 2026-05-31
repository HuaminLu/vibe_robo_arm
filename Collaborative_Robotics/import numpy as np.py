import numpy as np

# Load your camera parameters
data = np.load("./camera_params.npz")

# Print the contents to see the calibration values
print("--- Camera Matrix ---")
print(data["camera_matrix"])