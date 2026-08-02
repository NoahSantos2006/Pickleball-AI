import json
from pathlib import Path
import numpy as np
import cv2
import time
import sys

from ball_tracker import BallTracker
BASE_DIR = Path(__file__).parent.parent

def get_coordinates_and_center(prediction: dict) -> tuple:

    x = prediction['x']
    y = prediction['y']
    w = prediction['width']
    h = prediction['height']

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    return [x1, y1, x2, y2], (x, y)

def run_predictions(COURT_POINTS_INPUT_FILE: Path, PREDICTIONS_INPUT_FILE: Path, BALL_TRACKING_OUTPUT_FILE: Path, ball_tracker: BallTracker):

    with open(COURT_POINTS_INPUT_FILE, "r") as f:
    
        print(f"Opening {COURT_POINTS_INPUT_FILE}")
        court_points = json.load(f)
        
    with open(PREDICTIONS_INPUT_FILE, "r") as f:

        print(f"Opening {PREDICTIONS_INPUT_FILE}")
        predictions_by_frame = json.load(f)

    for frame_number, predictions_array in predictions_by_frame.items():

        print(f"Looking through frame {frame_number} out of {len(predictions_by_frame)}")
        current_ball_locations = []
    
        for pred in predictions_array:

            if pred['class'] == 'ball':

                coordinates, center = get_coordinates_and_center(prediction=pred)
                print(f"Current ball coordinates: {center}")
                current_ball_locations.append(center)

        ball_tracker.update(
            frame_number=int(frame_number), 
            ball_locations=current_ball_locations
        )

    with open(BALL_TRACKING_OUTPUT_FILE, "w") as f:
    
        print(f"Opening {BALL_TRACKING_OUTPUT_FILE}")
        json.dump(ball_tracker.tracker, f, indent=4)

def find_velocities(ball_track: dict, fps: int = 30) -> dict:

    frame_id = 1
    velocity = {}

    # velocities calcaulted in pixels/second

    while frame_id < len(ball_track):

        x, y = ball_track[frame_id]['vision model location']

        if frame_id == 1:

            vel_x, vel_y = 0, 0

            velocity[frame_id] = (vel_x, vel_y)
        
        else:

            prev_x, prev_y = ball_track[frame_id - 1]['vision model location']
 
            vel_x = (x - prev_x) / (1 / fps)
            vel_y = (y - prev_y) / (1 / fps) 
            
            velocity[frame_id] = (vel_x, vel_y)

        frame_id += 1

    return velocity

COURT_HEIGHT = 44
COURT_WIDTH = 20

SCALE = 20
PADDING = 50

def compute_homography(video_points: np.array, top_down_points = np.array([
                                                                [PADDING, PADDING + 44 * SCALE],                # near left baseline          
                                                                [PADDING + 20 * SCALE, PADDING + 44 * SCALE],   # near right baseline
                                                                [PADDING + 20 * SCALE, PADDING],                # far right baseline
                                                                [PADDING, PADDING],                             # far left baseline
                                                                [PADDING, PADDING + 29 * SCALE],                # near left kitchen
                                                                [PADDING + 20 * SCALE, PADDING + 29 * SCALE],   # near right kitchen
                                                                [PADDING, PADDING + 15 * SCALE],                # far right kitchen
                                                                [PADDING + 20 * SCALE, PADDING + 15 * SCALE]],  # far left kitchen
                                                                dtype=np.float32)
) -> np.array:
    
    H, mask = cv2.findHomography(
        video_points,
        top_down_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )

    if H is None: raise RuntimeError("Homography could not be computed")

    # Homography Matrix
    return H

def get_court_points(VIDEO_PATH: Path, OUTPUT_PATH: Path):

    # Click these landmarks in this exact order.
    POINT_NAMES = [
        "near_left_baseline",
        "near_right_baseline",
        "far_right_baseline",
        "far_left_baseline",
        "near_left_kitchen",
        "near_right_kitchen",
        "far_right_kitchen",
        "far_left_kitchen",
    ]

    clicked_points: list[list[int]] = []
    display_frame = None
    original_frame = None


    def mouse_callback(event, x, y, flags, param):
        
        global display_frame

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(clicked_points) >= len(POINT_NAMES):
            return

        clicked_points.append([x, y])

        point_number = len(clicked_points)
        point_name = POINT_NAMES[point_number - 1]

        print(f"{point_number}. {point_name}: ({x}, {y})")

        display_frame = original_frame.copy()

        for index, point in enumerate(clicked_points):
            px, py = point

            cv2.circle(
                display_frame,
                (px, py),
                6,
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                display_frame,
                str(index + 1),
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    def get_frame(video_path: str, frame_number: int = 0):

        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = capture.read()
        capture.release()

        if not success:
            raise RuntimeError(f"Could not read frame {frame_number}")

        return frame

    def save_points():
        data = {
            "video": VIDEO_PATH,
            "point_order": POINT_NAMES,
            "image_points": {
                name: point
                for name, point in zip(POINT_NAMES, clicked_points)
            },
        }

        Path(OUTPUT_PATH).write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

        print(f"Saved court points to {OUTPUT_PATH}")


    original_frame = get_frame(VIDEO_PATH, frame_number=0)
    display_frame = original_frame.copy()

    window_name = "Click court points"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Click the court points in this order:")

    for index, name in enumerate(POINT_NAMES, start=1):
        print(f"{index}. {name}")

    print("\nControls:")
    print("S = save")
    print("U = undo last point")
    print("R = reset all points")
    print("Q = quit")

    while True:
        frame_to_show = display_frame.copy()

        next_index = len(clicked_points)

        if next_index < len(POINT_NAMES):
            instruction = f"Click: {POINT_NAMES[next_index]}"
        else:
            instruction = "All points selected. Press S to save."

        cv2.putText(
            frame_to_show,
            instruction,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow(window_name, frame_to_show)

        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("u"):
            if clicked_points:
                removed = clicked_points.pop()
                print(f"Removed point: {removed}")

                display_frame = original_frame.copy()

                for index, point in enumerate(clicked_points):
                    px, py = point
                    cv2.circle(display_frame, (px, py), 6, (0, 255, 0), -1)
                    cv2.putText(
                        display_frame,
                        str(index + 1),
                        (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

        elif key == ord("r"):
            clicked_points.clear()
            display_frame = original_frame.copy()
            print("All points reset.")

        elif key == ord("s"):
            if len(clicked_points) != len(POINT_NAMES):
                print(
                    f"Select all {len(POINT_NAMES)} points before saving. "
                    f"Currently selected: {len(clicked_points)}"
                )
            else:
                save_points()
                break

    cv2.destroyAllWindows()
