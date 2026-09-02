import cv2
import json
import os

if __name__ == "__main__":

    video_file = os.path.join("input", "tennis.mp4")
    court_points_file = os.path.join("output", "court_points", "tennis_court_points_from_model.json")
    with open(court_points_file, "r") as f:

        court_points = json.load(f)

    cap = cv2.VideoCapture(video_file)

    cv2.namedWindow("First Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("First Frame", width=1920, height=1080)

    ret, frame = cap.read()

    if ret:

        for pred in court_points['1']:

            location = (int(pred['x']), int(pred['y']))
            print(location)
            cv2.circle(
                img=frame,
                center=location,
                radius=10,
                color=(0, 0, 0),
                thickness=3
            )
            cv2.putText(
                frame,
                f"{pred['class']}",
                location,
                fontScale=3,
                color=(0, 0, 0),
                thickness=3,
                fontFace=cv2.FONT_HERSHEY_COMPLEX
            )
        
        cv2.imshow("First Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optional: save it
        cv2.imwrite("first_frame.jpg", frame)

    else:

        print("Could not read video")