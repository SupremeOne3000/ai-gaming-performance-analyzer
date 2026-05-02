import cv2
from ultralytics import YOLO
from reaction_time import detect_shot
from crosshair_tracker import movement_score
import pandas as pd
import os
os.makedirs("output", exist_ok=True)

video_path = r"videos\MedalTVValorant20260412163927146.mp4"

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(video_path)

prev_frame = None
enemy_time = None

reaction_times = []
movement_scores = []

# METRICS
total_frames = 0
enemy_detections = 0
shot_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1
    print(f"Frame {total_frames} running...")

    results = model(frame, conf=0.25)

    # 🔥 NEW: track if enemy detected in THIS frame
    enemy_detected_this_frame = False

    for box in results[0].boxes:
        cls = int(box.cls[0])

        if cls == 0:  # person class
            enemy_detected_this_frame = True
            enemy_detections += 1

    if enemy_detected_this_frame and enemy_time is None:
        enemy_time = cap.get(cv2.CAP_PROP_POS_MSEC)

    if prev_frame is not None:
        if detect_shot(prev_frame, frame) and enemy_time is not None:
            shot_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            reaction = shot_time - enemy_time

            if 50 < reaction < 2000:
                reaction_times.append(reaction)

            shot_count += 1
            enemy_time = None  # reset after shot
        movement = movement_score(prev_frame, frame)
        movement_scores.append(movement)

    prev_frame = frame

cap.release()

avg_reaction = sum(reaction_times)/len(reaction_times) if reaction_times else 0
best_reaction = min(reaction_times) if reaction_times else 0
worst_reaction = max(reaction_times) if reaction_times else 0

avg_movement = sum(movement_scores)/len(movement_scores) if movement_scores else 0
max_movement = max(movement_scores) if movement_scores else 0
min_movement = min(movement_scores) if movement_scores else 0

data = [
    ["Total Frames Processed", total_frames],
    ["Enemy Detections", enemy_detections],
    ["Shots Detected", shot_count],

    ["", ""],
    ["--- Reaction Time ---", ""],
    ["Average Reaction Time (ms)", round(avg_reaction, 2)],
    ["Best Reaction Time (ms)", round(best_reaction, 2)],
    ["Worst Reaction Time (ms)", round(worst_reaction, 2)],

    ["", ""],
    ["--- Movement Analysis ---", ""],
    ["Average Movement Score", round(avg_movement, 2)],
    ["Max Movement", round(max_movement, 2)],
    ["Min Movement", round(min_movement, 2)]
]

df = pd.DataFrame(data, columns=["Metric", "Value"])
df.to_csv("output/report.csv", index=False)

print("\n" + "="*45)
print("      AI GAMING PERFORMANCE REPORT")
print("="*45)

print(f"\nTotal Frames Processed     : {total_frames}")
print(f"Enemy Detections          : {enemy_detections}")
print(f"Shots Detected            : {shot_count}")

print("\n--- Reaction Time ---")
print(f"Average Reaction Time     : {avg_reaction:.2f} ms")
print(f"Best Reaction Time        : {best_reaction:.2f} ms")
print(f"Worst Reaction Time       : {worst_reaction:.2f} ms")

print("\n--- Movement Analysis ---")
print(f"Average Movement Score    : {avg_movement:.2f}")
print(f"Maximum Movement          : {max_movement:.2f}")
print(f"Minimum Movement          : {min_movement:.2f}")

print("\n" + "="*45)
print("\n✅ Report saved in output/report.csv")