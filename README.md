# Vibe Robo Arm

A simple PyTorch + YOLOv5 computer vision demo for controlling a robot hand using webcam object detection.

## Overview

This project uses `torch.hub` to load a pretrained YOLOv5 model and `opencv-python` to capture webcam frames. Detected objects are mapped to basic hand control actions in `hand_control.py`.

## Files

- `cv.py` - object detection helper using YOLOv5 and OpenCV drawing utilities
- `hand_control.py` - webcam loop, inference, action mapping, and robot hand action stub

## Requirements

- Python 3.8+
- `torch`
- `opencv-python`

## Install

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install opencv-python
```

> If you do not have CUDA, install the CPU-only PyTorch wheel instead:
>
> ```bash
> python -m pip install torch torchvision torchaudio
> ```

## Run

```bash
python hand_control.py
```

Press `q` to exit the webcam window.

## Notes

- `hand_control.py` currently prints the selected action to the console. Replace `control_robot_hand()` with your actual robot hand interface.
- The action mapping in `ROBOT_ACTIONS` is a simple example. Update labels and actions to match your robot and application.

## Project inspiration

This repository is a small prototype for webcam-based object-aware robot hand control, suitable as a starting point for a robotics/innovation challenge project.
