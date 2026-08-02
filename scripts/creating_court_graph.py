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

if __name__ == "__main__":

    draw_pickleball_court()
