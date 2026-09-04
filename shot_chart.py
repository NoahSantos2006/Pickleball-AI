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

from scripts.side_functions import run_predictions, get_court_points, find_angles, compute_homography
from scripts.ball_tracker import BallTracker
from scripts.homography import live_homography_graph, shot_chart
from points import points

if __name__ == "__main__":

    video_filename = "pctennis4"
    sport = 'tennis'
    debug=False

    API_KEY = os.getenv("API_KEY")
    VIDEO_FILE = Path(os.path.join("input", f"{video_filename}.mp4"))
    BALL_TRACKING_CLASS_FILE = os.path.join("output", "BallTracking", f"{video_filename}", f"{video_filename}_ball_tracking_class.pkl")
    if sport == "pickleball":
        COURT_POINTS_FILE = os.path.join("output", "court_points", f"{video_filename}_court_points.json")
    elif sport == "tennis":
        COURT_POINTS_FILE = os.path.join("output", "court_points", f"{video_filename}_court_points_from_model.json")
    PREDICTIONS_FILE = os.path.join("output", "predictions", f"{video_filename}_predictions.txt")
    ACTUAL_BOUNCES_FILE = os.path.join("output", "BallTracking", f"{video_filename}", f"{video_filename}_actual_bounces.txt")

    shot_chart(COURT_POINTS_INPUT_FILE=COURT_POINTS_FILE, 
               PREDICTIONS_INPUT_FILE=PREDICTIONS_FILE, 
               BALL_TRACKING_CLASS_FILE=BALL_TRACKING_CLASS_FILE,
               VIDEO_FILE=VIDEO_FILE,
               debug=debug,
               graph_trajectory=False,
               SPORT=sport
            )
    

    



