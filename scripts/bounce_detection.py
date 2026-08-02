import json
from pathlib import Path
import os
import cv2
import sys
import numpy as np


from side_functions import find_velocities, get_coordinates_and_center, run_predictions

BASE_DIR = Path(__file__).parent.parent

def find_angles(ball_tracker: dict) -> list:

    frame_id = 2
    angles = {}

    while frame_id < len(ball_tracker) - 1:

        print(f"Finding angles for frame {frame_id}")

        # find locations
        x1, y1 = ball_tracker[frame_id - 1]['vision model location']
        x2, y2 = ball_tracker[frame_id]['vision model location']
        x3, y3 = ball_tracker[frame_id + 1]['vision model location']

        # movement vectors
        v1 = np.array([x2 - x1, y2 - y1])
        v2 = np.array([x3 - x2, y3 - y2])

        # normalize vectors in case they have different lengths (only care about direction)
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        # compute dot product to measure similarity between directions
        dot = np.clip(np.dot(v1, v2), -1, 1)

        # convert dot product into angle
        angle = np.arccos(dot)      # radians

        # convert to degrees
        angle_deg = np.degrees(angle)

        # save angle
        angles[frame_id] = angle_deg

        frame_id += 1

    return angles
    
if __name__ == "__main__":

    video_filename = "8"

    VIDEO_FILE = os.path.join(BASE_DIR, "input", f"{video_filename}.mp4")
    COURT_POINTS_INPUT_FILE = os.path.join(BASE_DIR, "output", "court_points", f"{video_filename}_court_points.json")
    PREDICTIONS_INPUT_FILE = os.path.join(BASE_DIR, "output", "predictions", f"{video_filename}_predictions.txt")
    BALL_TRACKING_FILE = os.path.join(BASE_DIR, "output", "BallTracking", f"{video_filename}_ball_tracking.json")

    with open(PREDICTIONS_INPUT_FILE, "r") as f:

        print(f"Opening {PREDICTIONS_INPUT_FILE}...")
        PREDICTIONS_PER_FRAME = json.load(f)

    with open(BALL_TRACKING_FILE, "r") as f:

        print(f"Opening {BALL_TRACKING_FILE}...")
        ball_tracker = json.load(f)
        ball_tracker = {int(keys): values for keys, values in ball_tracker.items()}

    # run_predictions(COURT_POINTS_INPUT_FILE=COURT_POINTS_INPUT_FILE, PREDICTIONS_INPUT_FILE=PREDICTIONS_INPUT_FILE)

    cap = cv2.VideoCapture(VIDEO_FILE)
    fps = cap.get(cv2.CAP_PROP_FPS)

    velocities = find_velocities(ball_track=ball_tracker, fps=fps)

    angles = find_angles(ball_tracker=ball_tracker)

    max_angle = float('-inf')
    for frame_id, angle in angles.items():

        print(f"Frame {frame_id}: {angle} degrees")

