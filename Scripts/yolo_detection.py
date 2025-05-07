import torch
import cv2

# Load YOLOv5 model (small version for faster processing)
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Change 'yolov5s' to 'yolov5m', 'yolov5l', 'yolov5x' for larger models
model_yolo.eval()

# Initialize webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()  # Read the frame from the webcam
    if not ret:
        break

    # Perform object detection on the frame
    results = model_yolo(frame)
    results.render()  # Render bounding boxes and labels on the frame

    # Show the frame with bounding boxes
    cv2.imshow("Detected Objects", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
