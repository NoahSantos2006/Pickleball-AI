import os
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import sys
import pickle

BASE_DIR = Path(__file__).parent.parent

from scripts.ball_tracker import BallTracker
from scripts.side_functions import get_coordinates_and_center, validate_bounce

"""
PICKLEBALL COURT MEASUREMENTS

                    20 ft
    <------------------------------->
   (0, 44)                           (20, 44)
    *-------------------------------*        -         -
    |               |               |        |         |
    |               |               |        |         |
    |               |               |        | 15 ft   |
    |               |               |        |         |
    |               |               |        |         |
    |(0, 29)        |               |(20, 29)|         |
    |-------------------------------*        -         |
    |                       -       |                  |
    |                       | 7 ft  |                  |
    |                       -       |                  |
    |-------------------------------* Net              | 44 ft
    |                               |                  |
    |                               |                  |
    |(0, 15)                        |(20, 15)          |
    |-------------------------------*                  |
    |               |               |                  |
    |               |               |                  |
    |               |               |                  |
    |               |               |                  |
    |               |               |                  |
    |               |               |                  |
    *-------------------------------*                  -
  (0, 0)                          (20, 0)
"""

"""

BALL TRACKER

Dictionary with:

    1. FRAME NUMBER
    2. BALL PREDICTIONS
    3. CHOSEN BALL LOCATION
    4. WHETHER IT'S AN ESTIMATION OR NOT
    5. TIES
        - if the last ball prediction had a tie between the amount of pixels between the previous detection than compare to the next one

    EXAMPLE:

    ball_tracker = {
        1: {
            ball_predictions: [
                {
                    homography_location: (x, y),
                    image_location: (x1, y1, x2, y2)
                }
            ]
            homography_location: (x, y)
            estimation: True
            ties: [(x1, y1), (x2, y2)]
        }
    }

Also add incorrect predictions like a light spot on the ground. The coordinates would be the same so we can rule out that detection

 - location predictions, if the x, y coordinate is the same then increase counter and a threshold will be met to rule out that detection for future predictions
 - might use x1, y1, x2, y2 for that so if the camera is still, then it's easier compared to homography

"""

"""

pickleball bouncing coordinates window

ex 1
    1.  [532.5, 701.5]
    2.  [504.0, 757.5]
    3.  [479.0, 767.0] --> bounced
    4.  [453.5, 767.0]
    5.  [430.0, 769.5]

ex 2
    1. [425.0, 307.0]
    2. [396.0, 326.5]
    3. [366.0, 347.5] --> bounced
    4. [347.0, 345.0]
    5. [328.5, 337.5]
    6. [310.5, 330.5]

ex 3
    1. [1345.0, 737.5]
    2. [1343.5, 781.5]
    3. [1343.0, 820.5] --> bounced
    4. [1346.0, 816.5]
    5. [1348.5, 806.5]

    

THINGS THAT I COULD IMPLEMENT

    - might implement avoiding detections that were in the same homographical spot for multiple frames meaning a false positive
"""

COURT_HEIGHT = 44
COURT_WIDTH = 20

SCALE = 20
PADDING = 50

def compute_homography(video_points: np.array, top_down_points = np.array([
                                                                [PADDING, PADDING + 44 * SCALE],                # near left baseline    ( 50 , 930 ) 
                                                                [PADDING + 20 * SCALE, PADDING + 44 * SCALE],   # near right baseline   ( 450, 930 )
                                                                [PADDING + 20 * SCALE, PADDING],                # far right baseline    ( 450, 50  )
                                                                [PADDING, PADDING],                             # far left baseline     ( 50 , 50  )
                                                                [PADDING, PADDING + 29 * SCALE],                # near left kitchen     ( 50 , 630 )
                                                                [PADDING + 20 * SCALE, PADDING + 29 * SCALE],   # near right kitchen    ( 450, 630 )
                                                                [PADDING, PADDING + 15 * SCALE],                # far right kitchen     ( 50 , 350 ) 
                                                                [PADDING + 20 * SCALE, PADDING + 15 * SCALE]],  # far left kitchen      ( 450, 350 )
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

def detection_of_court_points(box: tuple, H: np.array) -> tuple:

    x1, y1, x2, y2 = box

    # Find the center of the x coordinate
    ground_x = (x1 + x2) / 2
    ground_y = y2

    video_point = np.array(
        [[[ground_x, ground_y]]],
        dtype=np.float32
    )

    court_point = cv2.perspectiveTransform(video_point, H)

    court_x, court_y = court_point[0, 0]

    return int(court_x), int(court_y)

def draw_pickleball_court() -> np.array:

    OUTPUT_WIDTH = int(COURT_WIDTH * SCALE)
    OUTPUT_HEIGHT = int(COURT_HEIGHT * SCALE)

    # 1. Create a black canvas (500x500 pixels, 3 color channels)
    image = np.full((OUTPUT_HEIGHT + int(PADDING*2), OUTPUT_WIDTH + int(PADDING*2), 3), (60, 110, 60), dtype="uint8")
    
    """
    Top size
    """

    # Far Side
    cv2.rectangle(
        image,
        (PADDING, PADDING + 15 * SCALE),                     
        (PADDING + 20 * SCALE, PADDING),                        
        (235, 160, 85),
        -1                                  # Thickness (positive number --> outline only; -1 --> fill entire rectangle)
    )


    # Kitchen
    cv2.rectangle(
        image,
        (PADDING, PADDING + 29 * SCALE),
        (PADDING + 20 * SCALE, PADDING + 15 * SCALE),
        (150, 60, 38),
        -1
    )

    # Near Side
    cv2.rectangle(
        image,
        (PADDING, PADDING + 44 * SCALE),
        (PADDING + 20 * SCALE, PADDING + 29 * SCALE),
        (235, 160, 85),
        -1
    )

    # Net
    cv2.line(
        image,
        (PADDING, PADDING + 22 * SCALE),
        (PADDING + 20 * SCALE, PADDING + 22 * SCALE),
        (0, 0, 0),
        3
    )

    # Far Kitchen Line
    cv2.line(
        image,
        (PADDING, PADDING + 15 * SCALE),
        (PADDING + 20 * SCALE, PADDING + 15 * SCALE),
        (255, 255, 255),
        3
    )

    # Near Kitchen Line
    cv2.line(
        image,
        (PADDING, PADDING + 29 * SCALE),
        (PADDING + 20 * SCALE, PADDING + 29 * SCALE),
        (255, 255, 255),
        3
    )

    # Top Baseline
    cv2.line(
        image,
        (PADDING, PADDING),
        (PADDING + 20 * SCALE, PADDING),
        (255, 255, 255),
        3 
    )

    # Left SideLine
    cv2.line(
        image,
        (PADDING, PADDING),
        (PADDING, PADDING + 44 * SCALE),
        (255, 255, 255),
        3
    )

    # Top Side Line Seperator
    cv2.line(
        image,
        (PADDING + 10 * SCALE, PADDING + 15 * SCALE),
        (PADDING + 10 * SCALE, PADDING),
        (255, 255, 255),
        3
    )

    # Bottom Baseline
    cv2.line(
        image,
        (PADDING, PADDING + 44 * SCALE),
        (PADDING + 20 * SCALE, PADDING + 44 * SCALE),
        (255, 255, 255),
        3
    )

    # Right Sideline
    cv2.line(
        image,
        (PADDING + 20 * SCALE, PADDING),
        (PADDING + 20 * SCALE, PADDING + 44 * SCALE),
        (255, 255, 255),
        3
    )

    # Near Side Line Seperator
    cv2.line(
        image,
        (PADDING + 10 * SCALE, PADDING + 44 * SCALE),
        (PADDING + 10 * SCALE, PADDING + 29 * SCALE),
        (255, 255, 255),
        3
    )

    return image
          
def live_homography_graph(
    COURT_POINTS_INPUT_FILE: Path,
    PREDICTIONS_INPUT_FILE: Path,
    BALL_TRACKING_CLASS_FILE: Path
):

    PICKLEBALL_COURT_IMAGE = draw_pickleball_court()

    with open(COURT_POINTS_INPUT_FILE, "r") as f:
        print(f"Opening {COURT_POINTS_INPUT_FILE}")
        court_points = json.load(f)

    with open(PREDICTIONS_INPUT_FILE, "r") as f:
        print(f"Opening {PREDICTIONS_INPUT_FILE}")
        predictions_by_frame = json.load(f)
        predictions_by_frame = {int(keys): vals for keys, vals in predictions_by_frame.items()}

    with open(BALL_TRACKING_CLASS_FILE, "rb") as f:
        print(f"Opening {BALL_TRACKING_CLASS_FILE}")
        ball_tracker = pickle.load(f)

    video_points = [
        coordinate
        for point_location, coordinate
        in court_points["image_points"].items()
    ]

    video_points = np.array(video_points, dtype=np.float32)

    HOMOGRAPHY_MATRIX = compute_homography(
        video_points=video_points
    )

    window_name = "Pickleball Court"

    ball_tracking_stats = ball_tracker.tracker

    if len(ball_tracking_stats) == 0:
        print("No frames were found.")
        return

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_trackbar_change(value):
        # The frame is drawn in the main loop.
        pass

    cv2.createTrackbar(
        "Frame",
        window_name,
        0,
        len(ball_tracking_stats) - 1,
        on_trackbar_change
    )

    current_frame_index = -1

    while True:

        frame_index = cv2.getTrackbarPos(
            "Frame",
            window_name
        )

        frame_number = 1
        # Only redraw when the slider changes.
        if frame_index != current_frame_index:

            current_frame_index = frame_index

            predictions_array = predictions_by_frame[frame_number]

            court_frame = PICKLEBALL_COURT_IMAGE.copy()
            ball_found = False

            cv2.putText(
                court_frame,
                f"Frame Number: {frame_number}",
                (1, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

            frame_ball_tracking = ball_tracking_stats.get(
                frame_number
            )

            if frame_ball_tracking is None:
                continue

            location = frame_ball_tracking.get(
                "homography location"
            )

            if location == (-1, -1):
                continue

            ball_found = True

            location = (
                int(round(location[0])),
                int(round(location[1]))
            )

            cv2.circle(
                court_frame,
                location,
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                court_frame,
                f"Ball",
                location,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                court_frame,
                f"Ball Coordinates: {location}",
                (1, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

            for pred in predictions_array:

                # Skip only additional ball detections.
                # Your original code skipped every remaining prediction
                # after finding the ball.
            
                location = None

                if pred["class"] == "ball": continue

                coordinates, center = get_coordinates_and_center(
                    prediction=pred
                )

                location = detection_of_court_points(
                    box=coordinates,
                    H=HOMOGRAPHY_MATRIX
                )

                if location is None:
                    continue

                location = (
                    int(round(location[0])),
                    int(round(location[1]))
                )

                cv2.circle(
                    court_frame,
                    location,
                    5,
                    (0, 0, 255),
                    -1
                )

                label_location = (
                    location[0],
                    max(location[1] - 10, 15)
                )

                cv2.putText(
                    court_frame,
                    pred["class"],
                    label_location,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            cv2.imshow(
                window_name,
                court_frame
            )

        key = cv2.waitKey(33) & 0xFF

        if key == ord("q"):
            break

        # Previous frame
        elif key == ord("a"):
            previous_index = max(
                0,
                current_frame_index - 1
            )

            cv2.setTrackbarPos(
                "Frame",
                window_name,
                previous_index
            )

        # Next frame
        elif key == ord("d"):
            next_index = min(
                len(ball_tracking_stats) - 1,
                current_frame_index + 1
            )

            cv2.setTrackbarPos(
                "Frame",
                window_name,
                next_index
            )

    cv2.destroyAllWindows()

def shot_chart(
    COURT_POINTS_INPUT_FILE: Path,
    PREDICTIONS_INPUT_FILE: Path,
    BALL_TRACKING_CLASS_FILE: Path,
    VIDEO_FILE: Path,
    bounce_angle_threshold: np.float64 = np.float64(40),
    player_angle_difference_threshold: np.float64 = np.float64(95),
    bounce_frame_cooldown: int = 5,
    testing: bool = True,
):

    cap = cv2.VideoCapture(VIDEO_FILE)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_FILE}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    PICKLEBALL_COURT_IMAGE = draw_pickleball_court()
    
    with open(COURT_POINTS_INPUT_FILE, "r") as f:
        print(f"Opening {COURT_POINTS_INPUT_FILE}")
        court_points = json.load(f)

    with open(PREDICTIONS_INPUT_FILE, "r") as f:
        print(f"Opening {PREDICTIONS_INPUT_FILE}")
        predictions_by_frame = json.load(f)
        predictions_by_frame = {int(keys): vals for keys, vals in predictions_by_frame.items()}

    with open(BALL_TRACKING_CLASS_FILE, "rb") as f:
        print(f"Opening {BALL_TRACKING_CLASS_FILE}")
        ball_tracker = pickle.load(f)

    video_points = [
        coordinate
        for point_location, coordinate
        in court_points["image_points"].items()
    ]

    video_points = np.array(video_points, dtype=np.float32)

    HOMOGRAPHY_MATRIX = compute_homography(
        video_points=video_points
    )

    frame_number = 1
    bounce_cooldown = 0

    while True:

        success, video_frame = cap.read()

        if not success:
            break

        court_image = PICKLEBALL_COURT_IMAGE.copy()

        current_ball_frame_stats = ball_tracker.tracker.get(frame_number)
        ball_location = current_ball_frame_stats.get("homography location")
        ball_angle = current_ball_frame_stats.get("angle")
        bounced = False
        closest_player_to_ball = None
        player_locations = []

        if not ball_angle:
            frame_number += 1
            continue

        if validate_bounce(
            frame_number=frame_number,
            ball_tracker=ball_tracker,
            bounce_angle_threshold=bounce_angle_threshold,
            bounce_frame_cooldown=bounce_frame_cooldown,
            testing=testing
        ):

            if bounce_cooldown == 0:

                print(f"Ball bounced on frame {frame_number}")
                bounced = True
                if ball_angle < player_angle_difference_threshold:
                    cv2.circle(
                        img=PICKLEBALL_COURT_IMAGE,
                        center=ball_location,
                        radius=5,
                        color=(0, 0, 0),
                        thickness=3
                    )

                bounce_cooldown = bounce_frame_cooldown


        if bounce_cooldown > 0 and not bounced: 

            bounce_cooldown -= 1


        for pred in predictions_by_frame[frame_number]:

            if pred['class'] == 'ball' or 'box' not in pred: continue

            coordinates, center = get_coordinates_and_center(prediction=pred)

            location = detection_of_court_points(
                box=np.array(pred['box'], dtype=np.float32),
                H=HOMOGRAPHY_MATRIX
            )

            player_locations.append((location, center))

            if ball_angle >= player_angle_difference_threshold and bounced:

                ballx, bally = ball_location
                playerx, playery = location

                current_pixels_away = np.hypot(playerx - ballx, playery - bally)
                
                if not closest_player_to_ball:

                    closest_player_to_ball = [location, current_pixels_away]

                else:

                    if current_pixels_away < closest_player_to_ball[1]:
                        closest_player_to_ball = [location, current_pixels_away]

            cv2.circle(
                img=court_image,
                center=location,
                radius=1,
                color=(0, 0, 0),
                thickness=1
            )

            cv2.putText(
                court_image,
                pred['class'],
                (
                    location[0],
                    max(location[1] - 10, 15)
                ),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        if ball_angle >= player_angle_difference_threshold and bounced:
            print(f"Ball hit by player, location = {closest_player_to_ball[0]}")
            cv2.circle(
                img=PICKLEBALL_COURT_IMAGE,
                center=closest_player_to_ball[0],
                radius=5,
                color=(0, 0, 0),
                thickness=3
            )

        cv2.putText(
            court_image,
            f"Frame: {frame_number}",
            (10, 25),
            cv2.FONT_HERSHEY_COMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        # Resize court image to match the video frame's height.
        video_height, video_width = video_frame.shape[:2]
        court_height, court_width = court_image.shape[:2]

        scale = video_height / court_height

        resized_court_width = int(court_width * scale)

        court_frame = cv2.resize(
            court_image,
            (resized_court_width, video_height)
        )

        # Combine horizontally.
        side_by_side = cv2.hconcat([
            video_frame,
            court_frame
        ])

        cv2.imshow(
            "Video and Pickleball Court",
            side_by_side
        )

        delay = max(1, int(1000 / fps))
        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q"):
            break

        frame_number += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":

    from side_functions import run_predictions, get_court_points, get_coordinates_and_center

    video_filename = "8"

    COURT_POINTS_INPUT_FILE = os.path.join(BASE_DIR, "output", "court_points", f"{video_filename}_court_points.json")
    if not os.path.isfile(COURT_POINTS_INPUT_FILE):

        VIDEO_PATH = os.path.join(BASE_DIR, "input", f"{video_filename}.mp4")
        OUTPUT_PATH = COURT_POINTS_INPUT_FILE

        get_court_points(VIDEO_PATH=VIDEO_PATH, OUTPUT_PATH=OUTPUT_PATH)

    with open(COURT_POINTS_INPUT_FILE, "r") as f:
    
        video_points = json.load(f)

    court_points = []
    for point_location, coordinates in video_points['image_points'].items():

        court_points.append(coordinates)

    PREDICTIONS_INPUT_FILE = os.path.join(BASE_DIR, "output", "predictions", f"{video_filename}_predictions.txt")
    BALL_TRACKING_FILE = os.path.join(BASE_DIR, "output", "BallTracking", f"{video_filename}_ball_tracking.json")

    H = compute_homography(video_points=np.array(court_points, dtype=np.float32))
    ball_tracker = BallTracker(homography_matrix=H)

    run_predictions(COURT_POINTS_INPUT_FILE=COURT_POINTS_INPUT_FILE, PREDICTIONS_INPUT_FILE=PREDICTIONS_INPUT_FILE, BALL_TRACKING_OUTPUT_FILE=BALL_TRACKING_FILE, ball_tracker=ball_tracker)



    