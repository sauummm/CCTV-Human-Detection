import cv2
import torch
import os
from ultralytics import YOLO
from datetime import datetime

# Load YOLO model
model_path = "/Users/saum/Desktop/CCTV Human Detection/runs/detect/train2/weights/best.pt"
model = YOLO(model_path)

# RTSP stream URL (Change this to your RTSP URL)
rtsp_url = "test-vid.mp4"

# Output folder for saving detections
output_folder = "detections"
os.makedirs(output_folder, exist_ok=True)

# Open video stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Perform inference
    results = model(frame)

    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])  # Class ID
            conf = float(box.conf[0])  # Confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box

            # Draw bounding box if the detected class is "human"
            if model.names[cls_id] == "human":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Human {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save the detected frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(output_folder, f"human_{timestamp}.jpg")
                cv2.imwrite(save_path, frame)

    # Show real-time detection
    cv2.imshow("Human Detection", frame)

    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()