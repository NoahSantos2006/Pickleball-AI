import numpy as np
import cv2
import sys

class BallTracker:

    def __init__(self, homography_matrix, max_consecutive_predictions=30, max_displacement_px=300):

        self.HOMOGRAPHY_MATRIX = homography_matrix                      # homography matrix
        self.MAX_CONSECUTIVE_ESTIMATIONS = max_consecutive_predictions  # max number of estimations the tracker has before going back and checking the locations again
        self.MAX_CONSECUTIVE_ESTIMATIONS_PADDING = 30                   # when looking through the frames for an estimation error it goes back this any frames
        self.MAX_ESTIMATIONS_IN_REDO = 4                                # when creating the new ball tracker there can be this amount of estimations before we realize we just lost the ball
        self.MAX_DISPLACEMENT_PX = max_displacement_px                  # number of pixels between detection a ball can travel through frames

        self.tracker = {}

        self.no_location_found = None                                   # for the beginning when we have no detections
        self.consecutive_no_detections = 0                              
        self.consecutive_estimations = 0
        self.is_estimation = True

    def estimate_ball_location(self, frame_number: int, predictions: list):

        if len(self.tracker) == 0:

            self.no_location_found = True
            chosen_ball_location = (-1, -1)

        elif len(self.tracker) == 1:

            chosen_ball_location = self.tracker[frame_number - 1]['vision model location']

        else:

            if self.tracker[frame_number - 1]['vision model location'] == (-1, -1) or self.tracker[frame_number - 2]['vision model location'] == (-1, -1):

                chosen_ball_location = (-1, -1)

            else:

                x1, y1 = self.tracker[frame_number - 2]['vision model location']
                x2, y2 = self.tracker[frame_number - 1]['vision model location']

                x_estimation = x2 + (x2 - x1)
                y_estimation = y2 + (y2 - y1)

                chosen_ball_location = (x_estimation, y_estimation)
                print(f"We estimated with x_estimation and y_estimation here.")

        self.consecutive_estimations += 1
        self.is_estimation = True

        if self.consecutive_estimations >= self.MAX_CONSECUTIVE_ESTIMATIONS:

            print(f"There were {self.MAX_CONSECUTIVE_ESTIMATIONS} consecutive estimations on the ball. Check Frame {frame_number - self.MAX_CONSECUTIVE_ESTIMATIONS - self.MAX_CONSECUTIVE_ESTIMATIONS_PADDING} - {frame_number}")
            start_frame, i = frame_number - self.MAX_CONSECUTIVE_ESTIMATIONS, 0
            temp_ball_tracker = BallTracker(homography_matrix=self.HOMOGRAPHY_MATRIX)
            print(f"Creating a new ball tracker for frames {start_frame} - {frame_number}")
            while start_frame < frame_number:

                temp_ball_tracker.update(frame_number=i, ball_locations=self.tracker[start_frame]['ball locations'])
                start_frame += 1
                i += 1

            estimation_counter = 0
            for _, frame_data in temp_ball_tracker.tracker.items():

                if frame_data['estimation']:

                    estimation_counter += 1

            if estimation_counter > self.MAX_ESTIMATIONS_IN_REDO:

                print(f"Lost the ball.")
                sys.exit(1)

            else:

                print(f"We found the actual ball.")
                start_frame -= i
                i = 0
                while start_frame < frame_number:

                    print(f"temp_ball_tracker.tracker[{i}] = {temp_ball_tracker.tracker[i]}")
                    print(f"Updating self.tracker[{start_frame}]...")
                    self.tracker[start_frame] = temp_ball_tracker.tracker[i]
                    print(f"self.tracker[{start_frame}] = {self.tracker[start_frame]}")
                    start_frame += 1
                    i += 1

                if not self.update(frame_number=frame_number, ball_locations=predictions):

                    self.is_estimation = False
                    chosen_ball_location = self.tracker[frame_number]['vision model location']

        return chosen_ball_location

    def compute_homographical_location(self, ball_location: tuple) -> tuple:

        ground_x, ground_y = ball_location

        video_point = np.array(
            [[[ground_x, ground_y]]],
            dtype=np.float32
        )

        court_point = cv2.perspectiveTransform(video_point, self.HOMOGRAPHY_MATRIX)

        court_x, court_y = court_point[0, 0]

        return int(court_x), int(court_y)

    def fix_no_detections(self, last_frame: int):

        if  (
                (len(self.tracker) == 1) or 
                (self.tracker[last_frame - 1]['vision model location'] == (-1, -1)) or 
                (self.tracker[last_frame]['vision model location'] == (-1, -1)) or
                (len(self.tracker) < 3)
            ): return

        cur_x, cur_y = self.tracker[last_frame]['vision model location']
        prev_x, prev_y = self.tracker[last_frame - 1]['vision model location']

        x_change = cur_x - prev_x
        y_change = cur_y - prev_y

        changing_index = last_frame - 2
        while changing_index in self.tracker and self.tracker[changing_index]['vision model location'] == (-1, -1):

            next_x, next_y = self.tracker[changing_index + 1]['vision model location'] 
            self.tracker[changing_index]['vision model location'] = (next_x - x_change, next_y - y_change)
            self.tracker[changing_index]['homography location'] = self.compute_homographical_location(self.tracker[changing_index]['vision model location'])

            print(f"Changed Frame {changing_index} from (-1, -1) to {self.tracker[changing_index]['vision model location']} and homography location to {self.tracker[changing_index]['homography location']}")

            changing_index -= 1

        print(f"We fixed all the (-1, -1) locations: {self.tracker}")
        self.no_location_found = False

    # ball_locations is an array of ball locations from the model detection (not homographical)
    def update(self, frame_number: int, ball_locations: list):

        self.is_estimation = False
        print(f"CURRENT FRAME: {frame_number}\n------------------")
        print(f"Current ball locations: {ball_locations}")
        if frame_number > 1:
            print(f"Previous ball location: {self.tracker[frame_number - 1]['vision model location']}")
        nearest_ball_location = {}
        prediction_indices = {}

        if len(ball_locations) == 0:

            # if we don't detect a ball we estimate using a simple slope
            self.consecutive_no_detections += 1
            chosen_ball_location = self.estimate_ball_location(frame_number=frame_number, predictions=ball_locations)

        elif len(ball_locations) == 1:

            """
            if there is only one ball location detected we check to see if it's plausible, else we estimate
            """
            self.consecutive_no_detections = 0
            # if we are on the first frame then just pick the first one
            if len(self.tracker) == 0:

                chosen_ball_location = ball_locations[0]

            else:

                prev_location = self.tracker[frame_number - 1]['vision model location']

                curr_x, curr_y = ball_locations[0]
                prev_x, prev_y = prev_location

                if prev_x == -1 and prev_y == -1:

                    chosen_ball_location = curr_x, curr_y

                elif prev_x - self.MAX_DISPLACEMENT_PX > curr_x or curr_x > prev_x + self.MAX_DISPLACEMENT_PX:
                    print(f"current x ({curr_x}) location is out of location padding\nprev_x = {prev_x} and location padding = {self.MAX_DISPLACEMENT_PX}")
                    chosen_ball_location = self.estimate_ball_location(frame_number=frame_number, predictions=ball_locations)

                elif prev_y - self.MAX_DISPLACEMENT_PX > curr_y  or curr_y > prev_y + self.MAX_DISPLACEMENT_PX: 
                    print(f"Current y location ({curr_y}) is out of location padding\nprev_x = {prev_x} and location padding = {self.MAX_DISPLACEMENT_PX}")
                    chosen_ball_location = self.estimate_ball_location(frame_number=frame_number, predictions=ball_locations)

                else:

                    chosen_ball_location = ball_locations[0]

        else:

            self.consecutive_no_detections = 0
            """
            simple tracker for multiple ball predictions: chose closest from last frame

            create a dictionary for each ball location and compare

            use px

            ex 
                prev ball location: (100, 100)

                preds = [(102, 103), (101, 105)]

                pred[0] = |102 - 100| + |103 - 100| = 2 + 3 = 5px away
                pred[1] = |101 - 100| + |105 - 100| = 1 + 5 = 6px away

                we keep a dictionary
            """

            prev_x, prev_y = self.tracker[frame_number - 1]['vision model location']

            print(f"There are multiple ball locations: ball locations = {ball_locations}")

            for index, ball_location in enumerate(ball_locations):

                cur_x, cur_y = ball_location
                pixels_away = abs(cur_x - prev_x) + abs(cur_y - prev_y)

                # if we haven't looked at a ball then just put it in to start
                print(f"Nearest ball location: {nearest_ball_location}\nCurrent pixels away: {pixels_away}")
                if len(nearest_ball_location) == 0:

                    nearest_ball_location = {
                        index: {
                            'ball location': ball_location,
                            'pixels away': pixels_away
                        }
                    }

                else:

                    for idx, pixel_data in nearest_ball_location.items():

                        if pixels_away < pixel_data['pixels away']:

                            nearest_ball_location = {
                                index: {
                                    'ball location': ball_location,
                                    'pixels away': pixels_away
                                }
                            }

                        elif pixels_away == nearest_ball_location:

                            nearest_ball_location[len(nearest_ball_location) + 1] = {
                                'ball location': ball_location,
                                'pixels away': pixels_away
                            }

                        break

            print(f"Nearest ball location after iterations: {nearest_ball_location}")
            # for simplicity we will chose the first ball detected, but for future frames we will try to fix
            for idx in nearest_ball_location.keys():

                chosen_ball_location = nearest_ball_location[idx]['ball location']
                break
            

        # in case we have ties for detections that are n pixels away
        if len(prediction_indices) > 1: ties = nearest_ball_location
        else: ties = []

        print(f"Chosen ball location: {chosen_ball_location}")
        if chosen_ball_location != (-1, -1):
            print(f"This is current chosen ball location: {chosen_ball_location}")
            homography_location = self.compute_homographical_location(ball_location=chosen_ball_location)
        else: 
            print(f"We changed frame {frame_number}'s homography location to (-1, -1)")
            homography_location = (-1, -1)

        if self.is_estimation is None: print(f"Estimation is None on frame {frame_number}")

        if self.is_estimation is False: self.consecutive_estimations = 0
        
        self.tracker[int(frame_number)] = {
            'ball locations': ball_locations,                       # all predictions found
            'homography location': homography_location,             # homographic location of ball
            'vision model location': chosen_ball_location,          # vision model location of ball
            'estimation': self.is_estimation,                       # whether the chosen location is an estimation
            'ties': ties                                            # if multiple detections are the same amount of pixels away we set ties and check future predictions to see which are most plausible
        }

        # we made it so if the starting frames don't detect a ball we mark it as (-1, -1) and now we want to fix it
        if self.no_location_found == True: self.fix_no_detections(last_frame=frame_number)

        print(f"------------------")

        return self.is_estimation
