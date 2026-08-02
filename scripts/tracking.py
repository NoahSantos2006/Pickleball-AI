import cv2
import os
import numpy as np
from pathlib import Path
import json

from side_functions import run_predictions
from homography import compute_homography
from ball_tracker import BallTracker

BASE_DIR = Path(__file__).parent.parent

if __name__ == "__main__":

    video_filename = "8"
    
    COURT_POINTS_INPUT_FILE = os.path.join(BASE_DIR, "output", "court_points", f"{video_filename}_court_points.json")

    PREDICTIONS_INPUT_FILE = os.path.join(BASE_DIR, "output", "predictions", f"{video_filename}_predictions.txt")
    BALL_TRACKING_OUTPUT_FILE = os.path.join(BASE_DIR, "output", "BallTracking", f"{video_filename}_ball_tracking.json")

    with open(COURT_POINTS_INPUT_FILE, "r") as f:

        video_points = json.load(f)

    court_points = []
    for point_location, coordinates in video_points['image_points'].items():

        court_points.append(coordinates)


    H = compute_homography(video_points=np.array(court_points, dtype=np.float32))
    ball_tracker = BallTracker(homography_matrix=H)

    run_predictions(COURT_POINTS_INPUT_FILE=COURT_POINTS_INPUT_FILE, PREDICTIONS_INPUT_FILE=PREDICTIONS_INPUT_FILE, BALL_TRACKING_OUTPUT_FILE=BALL_TRACKING_OUTPUT_FILE, ball_tracker=ball_tracker)