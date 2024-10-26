import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO
import json

# Load YOLO model
model = YOLO("models/sunxds_0.4.1.engine")  # Path to your TensorRT model

def capture_screen():
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def detect_targets():
    frame = capture_screen()
    results = model(frame)
    targets = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if conf > 0.5:
                targets.append({
                    "class": cls,
                    "confidence": conf,
                    "center": [(x1 + x2) // 2, (y1 + y2) // 2]
                })
    return json.dumps(targets)

if __name__ == "__main__":
    print(detect_targets())
