import cv2
import torch
import json
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('models/sunxds_0.4.1.pt')  # Adjust path if necessary

def detect_targets(frame, fov_center, fov_radius):
    results = model(frame)
    targets = []

    for result in results:
        for box in result.boxes:
            if box.cls == 0:  # Class ID for 'person'
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                distance = ((center_x - fov_center[0]) ** 2 + (center_y - fov_center[1]) ** 2) ** 0.5

                if distance <= fov_radius:
                    targets.append({"x": center_x, "y": center_y})

    return targets

if __name__ == "__main__":
    fov_center = (960, 540)  # Center for 1920x1080 resolution
    fov_radius = 150

    cap = cv2.VideoCapture(0)  # Webcam capture (adjust as needed)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        targets = detect_targets(frame, fov_center, fov_radius)
        print(json.dumps(targets))  # Outputs JSON data for each frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
