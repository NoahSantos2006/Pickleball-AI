import cv2
import json
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import VideoFileSource, StreamConfig, VideoMetadata
from ultralytics import YOLO
import json
import supervision as sv
import numpy as np
from pathlib import Path
import sys
import base64
import pickle
from tqdm import tqdm

from dotenv import load_dotenv
import os
import time

from scripts.side_functions import run_predictions, get_court_points, find_angles, compute_homography, ball_near_player
from scripts.ball_tracker import BallTracker
from scripts.homography import live_homography_graph, shot_chart
from points import points

load_dotenv()

def create_directories():

    INPUT_PATH = Path("input")
    INPUT_PATH.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH = Path("output")
    ANNOTATED_OUTPUT_PATH = OUTPUT_PATH / "annotated_videos"
    COURT_POINTS_OUTPUT_PATH = OUTPUT_PATH / "court_points"
    PREDICTIONS_OUTPUT_PATH = OUTPUT_PATH / "predictions"
    BALL_TRACKING_OUTPUT_PATH = OUTPUT_PATH / "BallTracking"
    ANNOTATED_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    BALL_TRACKING_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    COURT_POINTS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

create_directories()

if __name__ == "__main__":

    video_filnames = ['laptop4', 'laptop6', 'laptop7', 'laptop8', 'laptop9', 'laptop10']
    video_filnames = ['tennis1', 'tennis2', 'tennis3']
    debug = True

    total_precision = 0   # tp / (tp + fp)
    total_recall = 0      # tp / (tp + fn)
    total_accuracy = 0    # (tp + tn) / (tp + tn + fp + fn)

    for video_filename in video_filnames:

        API_KEY = os.getenv("API_KEY")
        VIDEO_FILE = Path(os.path.join("input", f"{video_filename}.mp4"))
        BALL_TRACKING_CLASS_FILE = os.path.join("output", "BallTracking", f"{video_filename}", f"{video_filename}_ball_tracking_class.pkl")
        COURT_POINTS_FILE = os.path.join("output", "court_points", f"{video_filename}_court_points.json")
        PREDICTIONS_FILE = os.path.join("output", "predictions", f"{video_filename}_predictions.txt")
        ACTUAL_BOUNCES_FILE = os.path.join("output", "BallTracking", f"{video_filename}", f"{video_filename}_actual_bounces.txt")
        BOUNCE_DEBUG_FILE = os.path.join("output", "BallTracking", f"{video_filename}", f"{video_filename}_bouncing_debugging.txt")
        DEBUG_PATH = os.path.join("output", "BallTracking", f"{video_filename}", f"{video_filename}_debug.txt")

        try:
            with open(DEBUG_PATH, "w") as f:
                pass
        except FileNotFoundError as error:

            print(f"Could not find file: {DEBUG_PATH}.")
            os._exit(1)

        with open(PREDICTIONS_FILE, "r") as file:

            predictions_by_frame = json.load(file)

        with open(BALL_TRACKING_CLASS_FILE, "rb") as f:

            ball_tracker = pickle.load(f)

        if debug:

            tp, fp, fn, ball_tracker_class = find_angles(
                ball_tracker_class=ball_tracker, 
                actual_bounces_path=ACTUAL_BOUNCES_FILE, 
                BOUNCE_DEBUG_PATH=BOUNCE_DEBUG_FILE,
                predictions_dict=predictions_by_frame,
                DEBUG_PATH=DEBUG_PATH,
                debug=debug
            )

            precision = len(tp) / (len(tp) + len(fp))   # tp / (tp + fp)
            recall = len(tp) / (len(tp) + len(fn))      # tp / (tp + fn)

            true_negatives = len(ball_tracker_class.tracker) - len(tp) - len(fp) - len(fn)
            accuracy = (len(tp) + true_negatives) / len(ball_tracker_class.tracker)         # (tp + tn) / (tp + tn + fp + fn)

            print(f"{video_filename}: PRECISION = {precision:.2f} | RECALL = {recall:.2f} | ACCURACY = {accuracy:.2f}")

        else:

            find_angles(
                ball_tracker_class=ball_tracker,
                BOUNCE_DEBUG_PATH=BOUNCE_DEBUG_FILE,
                predictions_dict=predictions_by_frame,
                DEBUG_PATH=DEBUG_PATH,
                debug=debug
            )
        
        

    