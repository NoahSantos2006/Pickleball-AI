import cv2
import json
import numpy as np
from pathlib import Path
import os

from scripts.side_functions import get_court_points

def points(VIDEO_PATH: Path):

    parts = VIDEO_PATH.parts
    VIDEO_FILENAME = parts[-1].split('.')[0]

    VIDEO_PATH = str(VIDEO_PATH)
    OUTPUT_PATH = f"output/court_points/{VIDEO_FILENAME}_court_points.json"

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

    video_filename = "laptop12"

    VIDEO_PATH = os.path.join("input", f"{video_filename}.mp4")
    OUTPUT_PATH = os.path.join("output", "court_points", f"{video_filename}_court_points.json")

    get_court_points(VIDEO_PATH=VIDEO_PATH, OUTPUT_PATH=OUTPUT_PATH)