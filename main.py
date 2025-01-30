import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture('cars.mp4')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"]  # Truncated for brevity
model = YOLO('yolov8l.pt')

mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise ValueError("Error: Could not load mask image.")
ret, frame = cap.read()
if not ret:
    raise ValueError("Error: Could not read frame from camera.")
mask = cv2.resize(mask, (frame.shape[1], frame.shape[0])).astype(np.uint8)

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [545, 600, 1800, 600]  # Detection line

vehicle_colors = {"car": (0, 255, 0), "truck": (255, 0, 0), "bus": (0, 255, 255), "motorbike": (255, 0, 255)}
totalCount = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    imgRegion = cv2.bitwise_and(frame, frame, mask=mask)
    result = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in result:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in vehicle_colors and conf > 0.3:
                detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

    resultTracker = tracker.update(detections)
    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultTracker:
        x1, y1, x2, y2, id = map(int, result)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        vehicle_color = vehicle_colors.get(currentClass, (255, 255, 255))

        cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=20, rt=2, colorR=vehicle_color)
        cvzone.putTextRect(frame, f'ID: {id}', (x1, y1 - 10), scale=1, thickness=2, colorT=(0, 0, 0),
                           colorR=vehicle_color)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 10 < cy < limits[1] + 10:
            if id not in totalCount:
                totalCount.append(id)
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cv2.rectangle(frame, (50, 50), (350, 150), (50, 50, 50), -1)
    cv2.putText(frame, f'Count: {len(totalCount)}', (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
    cv2.putText(frame, "Press 'Q' to Quit", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Vehicle Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()