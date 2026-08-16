import json
from pathlib import Path
import os
import cv2
import sys
import numpy as np


from side_functions import find_velocities, get_coordinates_and_center, run_predictions

BASE_DIR = Path(__file__).parent.parent

def find_angles(ball_tracker: dict, bounce_angle_threshold: np.float64 = np.float64(40), angle_padding: int = 5) -> list:

    frame_id = 2
    angles = {}
    cur_pad = 0

    while frame_id < len(ball_tracker) - 1:

        # find locations
        x1, y1 = ball_tracker[frame_id - 1]['vision model location']
        x2, y2 = ball_tracker[frame_id]['vision model location']
        x3, y3 = ball_tracker[frame_id + 1]['vision model location']

        # movement vectors
        v1 = np.array([x2 - x1, y2 - y1])
        v2 = np.array([x3 - x2, y3 - y2])

        # normalize vectors in case they have different lengths (only care about direction)

        v1norm = np.linalg.norm(v1)
        if v1norm > 0:
            v1 = v1 / v1norm
        else:
            v1 = np.zeros_like(v1)

        v2norm = np.linalg.norm(v2)
        if v2norm > 0:
            v2 = v2 / v2norm
        else:
            v2 = np.zeros_like(v2)

        # compute dot product to measure similarity between directions
        dot = np.clip(np.dot(v1, v2), -1, 1)

        # convert dot product into angle
        angle = np.arccos(dot)      # radians
        
        # convert to degrees
        angle_deg = np.degrees(angle)

        # save angle
        angles[frame_id] = {
            'angle': angle_deg
        }

        ball_tracker[frame_id]['angle'] = angle_deg
        
        if angle_deg > bounce_angle_threshold:

            if cur_pad == 0:
                print(f"The ball bounced on frame {frame_id} with an angle of {angle_deg}")
                cur_pad = angle_padding

        if cur_pad > 0: cur_pad -= 1

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

    angles = find_angles(ball_tracker=ball_tracker)


