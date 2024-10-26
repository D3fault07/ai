import time
import yolo_aimbot

while True:
    detections = yolo_aimbot.detect_targets()
    print(detections)
    time.sleep(0.1)  # Adjust detection frequency if necessary
