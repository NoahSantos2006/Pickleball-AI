import json
import os
from pathlib import Path
import sys 
import numpy as np
import supervision as sv
import pickle

from tracking import BallTracker
from side_functions import compute_homography, run_predictions, BASE_DIR



if __name__ == "__main__":

    video_filename = "12"
    COURT_POINTS_INPUT_FILE = os.path.join(BASE_DIR, "output", "court_points", f"{video_filename}_court_points.json")
    PREDICTIONS_INPUT_FILE = os.path.join(BASE_DIR, "output", "predictions", f"{video_filename}_predictions.txt")
    BALL_TRACKING_CLASS_FILE = os.path.join(BASE_DIR, "output", "BallTracking", f"{video_filename}_ball_tracking_class.pkl")

    H = compute_homography(video_points=COURT_POINTS_INPUT_FILE)

    with open(BALL_TRACKING_CLASS_FILE, "rb") as f:

        ball_tracker = pickle.load(f)

    print(ball_tracker.tracker)