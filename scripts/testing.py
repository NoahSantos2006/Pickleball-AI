import json
import os
from pathlib import Path
import sys 
import numpy as np
import supervision as sv

from tracking import BallTracker
from side_functions import compute_homography, run_predictions, BASE_DIR

if __name__ == "__main__":
    
    COURT_POINTS_INPUT_FILE = os.path.join(BASE_DIR, "output", "court_points", "8_court_points.json")
    PREDICTIONS_INPUT_FILE = os.path.join(BASE_DIR, "output", "predictions", "8_predictions.txt")

    H = compute_homography(video_points=COURT_POINTS_INPUT_FILE)
    ball_tracker = BallTracker(homography_matrix=H)

    run_predictions(COURT_POINTS_INPUT_FILE=COURT_POINTS_INPUT_FILE, PREDICTIONS_INPUT_FILE=PREDICTIONS_INPUT_FILE, ball_tracker=BallTracker)