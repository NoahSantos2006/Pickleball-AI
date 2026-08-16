import json
from pathlib import Path
import numpy as np
import cv2
import time
import sys
import os

from scripts.ball_tracker import BallTracker

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

def validate_bounce(
        frame_number: int, 
        ball_tracker: BallTracker, 
        bounce_angle_threshold: np.float64, 
        bounce_frame_cooldown: int, 
        bounce_window: int = 3,
        slowdown_velocity_threshold: int = 20,
        consecutive_non_degrees_threshold: int = 3,
        testing: bool = True
    ):

    current_ball_frame_stats = ball_tracker.tracker.get(frame_number)
    next_ball_frame_stats = ball_tracker.tracker.get(frame_number + 1, None)

    if (
        (frame_number == len(ball_tracker.tracker)) or
        ('velocity' not in current_ball_frame_stats) or
        ('velocity' not in next_ball_frame_stats)
    ): return False

    frame_window = []
    curr_frame, end_frame = max(frame_number - bounce_window, 1), min(frame_number + bounce_window, len(ball_tracker.tracker))

    if not testing:
        print(f"{curr_frame} to {end_frame}", end=" ")

    consecutive_non_degrees = 0

    # if the ball wasn't found and it's suddenly found, the tracker gets a spike in location which causes a faulty detection
    """
    example:
        Frame 96 ball is 0.00 degrees at velocity (5.5, -7.5)
        Frame 97 ball is 0.00 degrees at velocity (5.5, -7.5)
        Frame 98 ball is 0.00 degrees at velocity (5.5, -7.5)
        Frame 99 ball is 148.32 degrees at velocity (5.5, -7.5)
        The ball bounced on frame 99.
        Frame 100 ball is 0.00 degrees at velocity (-2.0, 25.0)
        Frame 101 ball is 0.00 degrees at velocity (-2.0, 25.0)
        Frame 102 ball is 0.00 degrees at velocity (-2.0, 25.0)
    """
    positive_velocities = []
    while curr_frame <= end_frame:

        cur_frame_stats = ball_tracker.tracker.get(curr_frame)
        temp_cur_angle = cur_frame_stats.get('angle', None)
        velocity = cur_frame_stats.get('velocity', None)

        # if velocity == None
        if not velocity:
            curr_frame += 1
            continue

        curr_vx, curr_vy = velocity

        if temp_cur_angle <= 0.1:
            consecutive_non_degrees += 1
        else: 
            consecutive_non_degrees = 0

        # if velo[0:3] < 0 and velo[3] > 0 and velo[4:7] < 0
        if curr_vy < 0: positive_velocities.append(False)
        else: positive_velocities.append(True)

        curr_frame += 1

    cur_bounce_angle = current_ball_frame_stats.get("angle")
    cur_bounce_velo = current_ball_frame_stats.get('velocity')

    next_bounce_angle = next_ball_frame_stats.get("angle")
    next_bounce_velo = next_ball_frame_stats.get('velocity')

    angle_change = False
    sign_change = False
    slowdown = False
    
    cur_vx, cur_vy = cur_bounce_velo
    next_vx, next_vy = next_bounce_velo
    
    if not testing:
        print(positive_velocities, f"curr_vy = {cur_vy}; next_vy = {next_vy} on frame {frame_number}")

    if cur_vy * next_vy < 0: sign_change = True
    if abs(next_vy - cur_vy) > slowdown_velocity_threshold: slowdown = True
    if (cur_bounce_angle > bounce_angle_threshold): angle_change = True

    if (
        sum(positive_velocities) == 1 and (next_vy > 0 or cur_vy > 0) or 
        sum(positive_velocities) == len(positive_velocities) - 1 and (next_vy < 0 or cur_vy < 0)
    ):

        return False

    if (
        angle_change and (sign_change or slowdown) and 
        consecutive_non_degrees < consecutive_non_degrees_threshold
    ): return True
    
    return False

def find_angles(
        ball_tracker_class: BallTracker, 
        bounce_angle_threshold: np.float64 = np.float64(40), 
        angle_padding: int = 5,
        actual_bounces_path: Path = None,
        testing: bool = True
    ) -> list:

    actual_bounces = []

    if actual_bounces_path:

        with open(actual_bounces_path, "r") as f:

            actual_bounces = set([int(line.strip()) for line in f])
            og_length = len(actual_bounces)
        
    ball_tracker = ball_tracker_class.tracker

    # cleaning dict since the keys turn into strings after json serialize
    for k, v in ball_tracker.items(): 
        if isinstance(k, str): 
            ball_tracker = {int(keys): vals for keys, vals in ball_tracker.items()}
        break

    frame_id = 1
    angles = {}
    cur_pad = 0
    total_false_positives = 0
    false_positives = []

    # bounce detection
    while frame_id < len(ball_tracker) - 1:
        
        if (
            frame_id < 2 or
            ball_tracker[frame_id - 1]['vision model location'] == (-1, -1) or
            ball_tracker[frame_id]['vision model location'] == (-1, -1) or
            ball_tracker[frame_id + 1]['vision model location'] == (-1, -1)

        ):

            ball_tracker[frame_id]['angle'] = float('-inf')
            frame_id += 1
            continue

        # find locations
        x1, y1 = ball_tracker[frame_id - 1]['vision model location']
        x2, y2 = ball_tracker[frame_id]['vision model location']
        x3, y3 = ball_tracker[frame_id + 1]['vision model location']

        vx = x2 - x1
        vy = y2 - y1

        # calculates magnitude of a 2D velocity vecotry (pythagorean theorem)
        speed = np.hypot(vx, vy)

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

        if not testing:
            print(f"Frame {frame_id} ball is {angle_deg:.2f} degrees at velocity ({vx}, {vy})", end=" ")

        ball_tracker[frame_id]['angle'] = angle_deg
        ball_tracker[frame_id]['velocity'] = (vx, vy)
        
        if validate_bounce(
            frame_number=frame_id,
            ball_tracker=ball_tracker_class,
            bounce_frame_cooldown=angle_padding,
            bounce_angle_threshold=bounce_angle_threshold,
            testing=testing
        ):

            false_positive = True
            if cur_pad == 0:
                
                for frame in [frame_id - 1, frame_id, frame_id + 1]:

                    if frame in actual_bounces:
                        actual_bounces.discard(frame)
                        false_positive = False
                        break

                if false_positive: 
                    false_positives.append(frame_id)
                    total_false_positives += 1

                print(f"\nThe ball bounced on frame {frame_id}.\n")
                cur_pad = angle_padding

        if cur_pad > 0: cur_pad -= 1

        frame_id += 1

    ball_tracker_class.tracker = ball_tracker

    if actual_bounces_path:

        score = og_length - len(actual_bounces)
        print(f"\nWe detected {score} / {og_length} bounces\n{total_false_positives} false positives -> {false_positives}\nWe still have {actual_bounces}")

    return ball_tracker_class

def run_predictions(COURT_POINTS_INPUT_FILE: Path, PREDICTIONS_INPUT_FILE: Path, BALL_TRACKING_OUTPUT_FILE: Path, ball_tracker: BallTracker) -> BallTracker:

    with open(COURT_POINTS_INPUT_FILE, "r") as f:
    
        print(f"Opening {COURT_POINTS_INPUT_FILE}")
        court_points = json.load(f)
        
    with open(PREDICTIONS_INPUT_FILE, "r") as f:

        print(f"Opening {PREDICTIONS_INPUT_FILE}")
        predictions_by_frame = json.load(f)

    for frame_number, predictions_array in predictions_by_frame.items():

        # print(f"Looking through frame {frame_number} out of {len(predictions_by_frame)}")
        current_ball_locations = []
    
        for pred in predictions_array:

            if pred['class'] == 'ball' and pred['confidence'] >= 0.5:

                coordinates, center = get_coordinates_and_center(prediction=pred)
                print(f"Current ball coordinates: {center}")
                current_ball_locations.append(center)

        ball_tracker.update(
            frame_number=int(frame_number), 
            ball_locations=current_ball_locations
        )

    ball_tracker = find_angles(ball_tracker_class=ball_tracker)

    return ball_tracker

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

def compute_homography(COURT_POINTS_PATH: Path, top_down_points = np.array([
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

    with open(COURT_POINTS_PATH, "r") as f:
    
        video_points = json.load(f)

    court_points = []
    for point_location, coordinates in video_points['image_points'].items():

        court_points.append(coordinates)
        
    court_points = np.array(court_points, dtype=np.float32)
    
    H, mask = cv2.findHomography(
        court_points,
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

if __name__ == "__main__":

    from ball_tracker import BallTracker