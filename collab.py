#!/usr/bin/env python3
"""Simple CLI to run Collaborative_Robotics utilities from the repo root."""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(__file__)
CR_PATH = os.path.join(ROOT, "Collaborative_Robotics")

SCRIPTS = {
    "calibrate": "calibrateCamera.py",
    "transform": "getTransformationMatrix.py",
    "pick": "pickCVBlock.py",
    "test": "testDobot.py",
}


def run_script(name):
    if name not in SCRIPTS:
        print("Unknown script", name)
        return 2
    script_path = os.path.join(CR_PATH, SCRIPTS[name])
    if not os.path.exists(script_path):
        print("Missing:", script_path)
        return 3
    print(f"Running {script_path}")
    return subprocess.call([sys.executable, script_path])


def main():
    p = argparse.ArgumentParser(description="Run Collaborative_Robotics utilities")
    p.add_argument("action", choices=list(SCRIPTS.keys()), help="which utility to run")
    args = p.parse_args()
    rc = run_script(args.action)
    sys.exit(rc)


if __name__ == "__main__":
    main()
