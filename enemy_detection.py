from ultralytics import YOLO
import cv2

def detect_enemies(video_path):

    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print("Frame running...")

        # Run YOLO with lower confidence (important for game footage)
        results = model(frame, conf=0.25)

        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            print(f"Detected class: {cls}, Confidence: {conf:.2f}")

            # Class 0 = person (COCO dataset)
            if cls == 0:
                print("Person detected!")

                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw rectangle manually
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Enemy {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Enemy Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()