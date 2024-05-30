# Bayazid-Khan-Burki
To develop the described Python script using OpenCV and other necessary libraries for RTSP video stream processing and object detection with YOLOv8, I will intend to follows these steps:
1.	Set Up the Environment:
o	Install necessary libraries: OpenCV, ultralytics (for YOLOv8), and others.
o	Ensure the RTSP stream is accessible.
2.	Fetch RTSP Stream:
o	Use OpenCV to capture the RTSP stream from the mobile camera.
3.	Object Detection:
o	Use YOLOv8 to detect multiple people in the stream.
o	Draw bounding boxes around detected people with unique colors, avoiding shades of red.
4.	User Interaction:
o	Show video in an OpenCV window.
o	Allow users to click on a bounding box to change its color to red and start a timer.
o	Ensure tracking of the person even if detection is temporarily lost.
o	Handle switching of red boxes and timers when a different person is selected.

import cv2
import numpy as np
import random
import time
from ultralytics import YOLO

#rtsp_cam =("http://192.168.18.148:8080/video")
#cap = cv2.VideoCapture(rtsp_cam)
# Initialize the YOLOv8 model
model = YOLO('yolov8n.pt')  # Assuming 'yolov8n.pt' is the model file

# Function to generate random colors avoiding red shades
def generate_color():
    while True:
        color = [random.randint(0, 255) for _ in range(3)]
        if not (color[0] > 200 and color[1] < 50 and color[2] < 50):
            return color

# Function to check if a point is inside a bounding box
def is_inside_box(box, x, y):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

# RTSP stream URL (replace with your RTSP stream URL)
rtsp_url = "http://192.168.18.148:8080/video"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Initialize variables
colors = {}
selected_person_id = None
start_time = None
tracker = None

# Mouse callback function
def select_person(event, x, y, flags, param):
    global selected_person_id, start_time, tracker
    if event == cv2.EVENT_LBUTTONDOWN:
        for person_id, (box, color) in colors.items():
            if is_inside_box(box, x, y):
                selected_person_id = person_id
                start_time = time.time()
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, tuple(box))
                break

# Set mouse callback function
cv2.namedWindow("RTSP Stream")
cv2.setMouseCallback("RTSP Stream", select_person)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Draw bounding boxes and assign colors
    detections = results[0].boxes.data.cpu().numpy()  # Adjusting to the correct attribute
    for detection in detections:
        x1, y1, x2, y2, conf, cls = map(int, detection[:6])
        if cls == 0:  # Assuming class 0 is person
            person_id = (x1, y1, x2, y2)
            if person_id not in colors:
                colors[person_id] = generate_color()

            color = colors[person_id]
            if selected_person_id == person_id:
                color = (0, 0, 255)  # Red color for selected person
                if tracker:
                    success, box = tracker.update(frame)
                    if success:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.putText(frame, f'Timer: {int(time.time() - start_time)}s', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Show the video frame
    cv2.imshow("RTSP Stream", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
