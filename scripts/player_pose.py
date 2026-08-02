import os
from pathlib import Path
import json
import cv2
import matplotlib.pyplot as plt
import sys

from ultralytics import YOLO


from side_functions import BASE_DIR

"""

Index	Body Part
0	    Nose
1	    Left Eye
2	    Right Eye
3	    Left Ear
4	    Right Ear
5	    Left Shoulder
6	    Right Shoulder
7	    Left Elbow
8	    Right Elbow
9	    Left Wrist
10	    Right Wrist
11	    Left Hip
12	    Right Hip
13	    Left Knee
14	    Right Knee
15	    Left Ankle
16	    Right Ankle

"""

if __name__ == "__main__":

    video_filename = "8"


    PLAYER_POSE_JSON_FILE = os.path.join(BASE_DIR, "output", "pose", f"{video_filename}_pose.json")
    PLAYER_POSE_VIDEO_FILE = os.path.join(BASE_DIR, "output", "pose", f"{video_filename}_pose.mp4")
    if os.path.isfile(PLAYER_POSE_JSON_FILE):

        INPUT_VIDEO = os.path.join(BASE_DIR, "input", f"{video_filename}.mp4")
        POSE_MODEL_PATH = os.path.join(BASE_DIR, "yolo11n-pose.pt")

        pose_model = YOLO(POSE_MODEL_PATH)

        # Reopen original video
        cap = cv2.VideoCapture(INPUT_VIDEO)
    
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        out = cv2.VideoWriter(
            PLAYER_POSE_VIDEO_FILE,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )
    
        frame_id = 1
        pose_dict = {}
        
        while True:
    
            ret, frame = cap.read()
            if not ret:
                break

            pose_results = pose_model(frame, conf=0.4, verbose=False, device='cpu')
            pose_result = pose_results[0]

            annotated_frame = pose_result.plot()

            print(f"Writing pose results to frame {frame_id}...")
            out.write(annotated_frame)

            keypoints = pose_result.keypoints
    
            if (
                keypoints is not None
                and keypoints.xy is not None
                and keypoints.conf is not None
                and len(keypoints.xy) > 0
            ):
                xy = pose_result.keypoints.xy.detach().cpu().numpy().tolist()
                conf = pose_result.keypoints.conf.detach().cpu().numpy().tolist()
    
                pose_dict[frame_id] = {
                    "xy": xy,
                    "confidence": conf
                }

            frame_id += 1

        cap.release()
        out.release()

        with open(PLAYER_POSE_JSON_FILE, "w") as f:

            print(f"Saving pose results to {PLAYER_POSE_JSON_FILE}...")
            json.dump(pose_dict, f, indent=4)

        print(f"Pose video saved to: {PLAYER_POSE_VIDEO_FILE}")