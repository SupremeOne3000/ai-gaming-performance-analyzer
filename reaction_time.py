import cv2
import numpy as np

def detect_shot(prev_frame, current_frame):

    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)

    mean_diff = np.mean(diff)

    print(f"Frame diff: {mean_diff:.2f}")  # debug

    if mean_diff > 15:   # slightly better threshold
        print("Shot detected!")
        return True

    return False