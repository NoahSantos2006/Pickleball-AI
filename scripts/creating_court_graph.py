import cv2
import numpy as np

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

def draw_pickleball_court():

    COURT_HEIGHT = 44
    COURT_WIDTH = 20

    SCALE = 20
    PADDING = 50

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

    # 3. Display the image in a window
    cv2.imshow("Pickleball Court", image)

    # 4. Keep window open until a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# IN METERS
def draw_tennis_court():

    # Tennis court dimensions in metres
    COURT_LENGTH = 23.77
    COURT_WIDTH = 10.97

    DOUBLES_ALLEY = 1.37
    SERVICE_LINE_FROM_NET = 6.40

    HALF_LENGTH = COURT_LENGTH / 2
    HALF_WIDTH = COURT_WIDTH / 2

    # Drawing settings
    SCALE = 25
    PADDING = 30

    # Vertical orientation:
    # width = court width
    # height = court length
    OUTPUT_WIDTH = int(COURT_WIDTH * SCALE)
    OUTPUT_HEIGHT = int(COURT_LENGTH * SCALE)

    image = np.full(
        (
            OUTPUT_HEIGHT + PADDING * 2,
            OUTPUT_WIDTH + PADDING * 2,
            3
        ),
        (60, 110, 60),
        dtype="uint8"
    )

    # Convert court coordinates in metres to pixels
    # x = court width
    # y = court length
    def point(x, y):
        return (
            int(PADDING + x * SCALE),
            int(PADDING + y * SCALE)
        )

    WHITE = (255, 255, 255)
    COURT_COLOR = (80, 150, 80)
    SERVICE_COLOR = (70, 140, 70)

    # -----------------------------------
    # Court background
    # -----------------------------------
    cv2.rectangle(
        image,
        point(0, 0),
        point(COURT_WIDTH, COURT_LENGTH),
        COURT_COLOR,
        -1
    )

    # -----------------------------------
    # Important Y positions
    # -----------------------------------
    top_service_y = HALF_LENGTH - SERVICE_LINE_FROM_NET
    bottom_service_y = HALF_LENGTH + SERVICE_LINE_FROM_NET

    # -----------------------------------
    # Service box area
    # -----------------------------------
    cv2.rectangle(
        image,
        point(DOUBLES_ALLEY, top_service_y),
        point(COURT_WIDTH - DOUBLES_ALLEY, bottom_service_y),
        SERVICE_COLOR,
        -1
    )

    # -----------------------------------
    # Outer doubles lines
    # -----------------------------------

    # Top baseline
    cv2.line(
        image,
        point(0, 0),
        point(COURT_WIDTH, 0),
        WHITE,
        3
    )

    # Bottom baseline
    cv2.line(
        image,
        point(0, COURT_LENGTH),
        point(COURT_WIDTH, COURT_LENGTH),
        WHITE,
        3
    )

    # Left doubles sideline
    cv2.line(
        image,
        point(0, 0),
        point(0, COURT_LENGTH),
        WHITE,
        3
    )

    # Right doubles sideline
    cv2.line(
        image,
        point(COURT_WIDTH, 0),
        point(COURT_WIDTH, COURT_LENGTH),
        WHITE,
        3
    )

    # -----------------------------------
    # Singles sidelines
    # -----------------------------------

    # Left singles sideline
    cv2.line(
        image,
        point(DOUBLES_ALLEY, 0),
        point(DOUBLES_ALLEY, COURT_LENGTH),
        WHITE,
        3
    )

    # Right singles sideline
    cv2.line(
        image,
        point(COURT_WIDTH - DOUBLES_ALLEY, 0),
        point(COURT_WIDTH - DOUBLES_ALLEY, COURT_LENGTH),
        WHITE,
        3
    )

    # -----------------------------------
    # Service lines
    # -----------------------------------

    # Top service line
    cv2.line(
        image,
        point(DOUBLES_ALLEY, top_service_y),
        point(COURT_WIDTH - DOUBLES_ALLEY, top_service_y),
        WHITE,
        3
    )

    # Bottom service line
    cv2.line(
        image,
        point(DOUBLES_ALLEY, bottom_service_y),
        point(COURT_WIDTH - DOUBLES_ALLEY, bottom_service_y),
        WHITE,
        3
    )

    # -----------------------------------
    # Centre service line
    # -----------------------------------
    cv2.line(
        image,
        point(HALF_WIDTH, top_service_y),
        point(HALF_WIDTH, bottom_service_y),
        WHITE,
        3
    )

    # -----------------------------------
    # Net
    # -----------------------------------
    cv2.line(
        image,
        point(0, HALF_LENGTH),
        point(COURT_WIDTH, HALF_LENGTH),
        (0, 0, 0),
        4
    )

    # -----------------------------------
    # Centre marks on baselines
    # -----------------------------------
    CENTER_MARK_LENGTH = 0.10

    # Top baseline centre mark
    cv2.line(
        image,
        point(HALF_WIDTH, 0),
        point(HALF_WIDTH, CENTER_MARK_LENGTH),
        WHITE,
        3
    )

    # Bottom baseline centre mark
    cv2.line(
        image,
        point(HALF_WIDTH, COURT_LENGTH),
        point(HALF_WIDTH, COURT_LENGTH - CENTER_MARK_LENGTH),
        WHITE,
        3
    )

    # -----------------------------------
    # Display court
    # -----------------------------------
    cv2.imshow("Vertical Tennis Court", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    draw_tennis_court()
