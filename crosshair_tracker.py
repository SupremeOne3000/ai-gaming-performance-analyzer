import cv2
import numpy as np

def movement_score(frame1, frame2):

    h, w, _ = frame1.shape

    # Focus on center region (crosshair area)
    cx, cy = w // 2, h // 2
    size = 100  # area size

    region1 = frame1[cy-size:cy+size, cx-size:cx+size]
    region2 = frame2[cy-size:cy+size, cx-size:cx+size]

    gray1 = cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)

    movement = np.mean(diff)

    print(f"Movement score: {movement:.2f}")  # debug

    return movement