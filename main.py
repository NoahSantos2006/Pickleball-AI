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

from dotenv import load_dotenv
import os
import time

load_dotenv()

def create_directories():

    INPUT_PATH = Path("input")
    INPUT_PATH.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH = Path("output")
    ANNOTATED_OUTPUT_PATH = OUTPUT_PATH / "annotated_videos"
    PREDICTIONS_OUTPUT_PATH = OUTPUT_PATH / "predictions"
    ANNOTATED_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

create_directories()

def predict_with_tracker(video_path: Path, api_key: str):

    client = InferenceHTTPClient.init(
        api_url="https://serverless.roboflow.com",
        api_key=api_key
    )

    parts = video_path.parts

    VIDEO_FILENAME = parts[-1].split(".")[0]

    # example: input/make/dunk/make4.mp4
    INPUT_VIDEO = str(video_path)

    # example: output/annotated_videos/make/dunk/make4_annotated.mp4
    OUTPUT_VIDEO = f"output/annotated_videos/{VIDEO_FILENAME}_annotated.mp4"

    source = VideoFileSource(INPUT_VIDEO, realtime_processing=False)

    config = StreamConfig(
        stream_output=[],
        data_output=["predictions"],
        requested_plan="webrtc-gpu-medium",
        requested_region="us",
    )

    session = client.webrtc.stream(
        source=source,
        workflow="pickleball-detection-vpickleball-detection-1sjz9-6-rfdetr-medium-t1-logic-2",
        workspace="noahs-workspace-kg24g",
        image_input="image",
        config=config
    )

    predictions_by_frame = {}

    @session.on_data()
    def on_data(data: dict, metadata: VideoMetadata):
        frame_id = metadata.frame_id

        # Adjust this if your workflow output structure is different
        preds = data.get("predictions", {}).get("predictions", [])

        predictions_by_frame[frame_id] = preds
        print(f"Saved predictions for frame {frame_id}: {len(preds)} detections")

    session.run()

    # Reopen original video
    cap = cv2.VideoCapture(INPUT_VIDEO)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = sv.ByteTrack(
        frame_rate=fps,
        track_activation_threshold=0.25,    # min confidence required for a detection to start a new track
        lost_track_buffer=30,               # how many consecutive frames ByteTrack will remember an object after it 
                                            # disappears before deleting its track
        minimum_matching_threshold=0.8,
        minimum_consecutive_frames=2        # how many consecutive detections are required before ByteTrack reports a track
    )

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_id = 1

    tracked_predictions = {}
    CLASS_NAMES = {
        0: "ball",
        1: "player"
    }


    while True:

        ret, frame = cap.read()
        if not ret:
            break

        preds = predictions_by_frame.get(frame_id, [])

        cache = {}

        xyxy = []
        confidences = []
        class_ids = []

        for pred in preds:

            class_name = pred['class']
            conf = pred['confidence']
            x = pred["x"]
            y = pred["y"]
            w = pred["width"]
            h = pred["height"]

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            player_box = [x1, y1, x2, y2]
            pred['box'] = [x1, y1, x2, y2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            xyxy.append(player_box)
            confidences.append(pred['confidence'])
            class_ids.append(pred['class_id'])

        tracked_predictions[frame_id] = []

        if xyxy:

            detections = sv.Detections(
                xyxy=np.array(xyxy, dtype=np.float32),
                confidence=np.array(confidences, dtype=np.float32),
                class_id=np.array(class_ids, dtype=int)
            )
            
        else: 
            detections = sv.Detections.empty()

        print(f"Updating tracker for frame {frame_id}. ({frame_id}/{len(predictions_by_frame)})")
        tracked = tracker.update_with_detections(detections)

        for box, confidence, class_id, tracker_id in zip(
            tracked.xyxy,
            tracked.confidence,
            tracked.class_id,
            tracked.tracker_id
        ):
            
            x1, y1, x2, y2 = map(int, box)
            curr_class = CLASS_NAMES[int(class_id)]

            tracked_predictions[frame_id].append({
                "class": curr_class,
                "track_id": None if tracker_id is None else int(tracker_id),
                "class_id": int(class_id),
                "confidence": float(confidence),
                "box": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ]
            })

            curr_conf = float(confidence)

            label = f"{curr_class} (tracker id: {tracker_id}) {curr_conf:.2f}"

            cv2.putText(
                frame,                      # image we're drawing on
                label,                      # text that will be displayed
                (x1, max(y1 - 10, 20)),     # bottom left corner of the text
                cv2.FONT_HERSHEY_SIMPLEX,   # font
                0.6,                        # font size
                (0, 255, 0),                # font color
                2                           # font thickness
            )

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

    predictions_text_path = f"output/predictions/{VIDEO_FILENAME}_predictions.txt"
    with open(predictions_text_path, "w") as f:

        print(f"Saving Predictions by Frame to {predictions_text_path}.")
        json.dump(predictions_by_frame, f, indent=4)

    tracker_path = f"output/predictions/{VIDEO_FILENAME}_tracker.txt"
    with open(tracker_path, "w") as f:

        print(f"Saving Tracker to {tracker_path}.")
        json.dump(tracked_predictions, f, indent=4)

    print(f"Done! Annotated video saved as output/annotated_videos/{'/'.join(parts[-3:-1])}/{VIDEO_FILENAME}_with_tracker_annotated.mp4")

    return len(predictions_by_frame) / fps

def predict(video_path: Path, api_key: str):

    client = InferenceHTTPClient.init(
        api_url="https://serverless.roboflow.com",
        api_key=api_key
    )

    parts = video_path.parts

    pose_model = YOLO("yolo11n-pose.pt")

    VIDEO_FILENAME = parts[-1].split(".")[0]

    # example: input/make/dunk/make4.mp4
    INPUT_VIDEO = str(video_path)

    # example: output/annotated_videos/make/dunk/make4_annotated.mp4
    OUTPUT_VIDEO = f"output/annotated_videos/{VIDEO_FILENAME}_annotated.mp4"

    source = VideoFileSource(INPUT_VIDEO, realtime_processing=False)

    config = StreamConfig(
        stream_output=[],
        data_output=["predictions"],
        requested_plan="webrtc-gpu-medium",
        requested_region="us",
    )

    session = client.webrtc.stream(
        source=source,
        workflow="pickleball-detection-vpickleball-detection-1sjz9-6-rfdetr-medium-t1-logic-2",
        workspace="noahs-workspace-kg24g",
        image_input="image",
        config=config
    )

    predictions_by_frame = {}

    @session.on_data()
    def on_data(data: dict, metadata: VideoMetadata):
        frame_id = metadata.frame_id

        # Adjust this if your workflow output structure is different
        preds = data.get("predictions", {}).get("predictions", [])

        predictions_by_frame[frame_id] = preds
        print(f"Saved predictions for frame {frame_id}: {len(preds)} detections")

    session.run()

    # Reopen original video
    cap = cv2.VideoCapture(INPUT_VIDEO)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
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

        print(f"Reading data for frame {frame_id}")
        pose_results = pose_model(frame, conf=0.4, verbose=False, device='cpu')
        pose_result = pose_results[0]
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

        cv2.putText(
            frame,
            f"Frame Number: {frame_id}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,   # font
            0.6,                        # font size
            (0, 255, 0),                # font color
            2 
        )

        preds = predictions_by_frame.get(frame_id, [])

        xyxy = []
        confidences = []
        class_ids = []

        for pred in preds:

            class_name = pred['class']
            conf = pred['confidence']
            x = pred["x"]
            y = pred["y"]
            w = pred["width"]
            h = pred["height"]

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            player_box = [x1, y1, x2, y2]
            pred['box'] = [x1, y1, x2, y2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            xyxy.append(player_box)
            confidences.append(pred['confidence'])
            class_ids.append(pred['class_id'])

            label = f"{class_name} ({conf}) ({x}, {y})"
            
            cv2.putText(
                frame,                      # image we're drawing on
                label,                      # text that will be displayed
                (x1, max(y1 - 10, 20)),     # bottom left corner of the text
                cv2.FONT_HERSHEY_SIMPLEX,   # font
                0.6,                        # font size
                (0, 255, 0),                # font color
                2                           # font thickness
            )

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

    predictions_text_path = f"output/predictions/{VIDEO_FILENAME}_predictions.txt"
    with open(predictions_text_path, "w") as f:

        print(f"Saving Predictions by Frame to {predictions_text_path}.")
        json.dump(predictions_by_frame, f, indent=4)

    print(f"Done! Annotated video saved as output/annotated_videos/{'/'.join(parts[-3:-1])}/{VIDEO_FILENAME}_annotated.mp4")

    pose_path = f"output/pose/{VIDEO_FILENAME}_pose.json"
    with open(pose_path, "w") as f:

        print(f"Saving YOLO Pose results in {pose_path}.")
        json.dump(pose_dict, f, indent=4)

    return len(predictions_by_frame) / fps

if __name__ == "__main__":

    API_KEY = os.getenv("API_KEY")
    INPUT_PATH = Path(os.path.join("input", "11.mp4"))

    predict(video_path=INPUT_PATH, api_key=API_KEY)



