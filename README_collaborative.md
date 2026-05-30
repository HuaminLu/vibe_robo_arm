# Collaborative_Robotics - Quick Runners

This repo root contains simple launchers to run the example scripts in the `Collaborative_Robotics` folder without changing directories.

Usage (from repository root):

Python (recommended):

```
python collab.py calibrate    # runs Collaborative_Robotics/calibrateCamera.py
python collab.py transform    # runs Collaborative_Robotics/getTransformationMatrix.py
python collab.py pick         # runs Collaborative_Robotics/pickCVBlock.py
python collab.py test         # runs Collaborative_Robotics/testDobot.py
```

Convenience single-file launchers are also provided:

```
python run_calibrate.py
python run_transform.py
python run_pick.py
python run_test_dobot.py
```

Notes:
- These wrappers simply execute the original scripts in `Collaborative_Robotics`.
- Hardware is required to run the Dobot-related scripts; if you don't have the robot connected, use the scripts for static inspection only.
- If you need me to add argument forwarding, better error handling, or a dry-run/hardware-emulation mode, tell me and I'll add it.
