import torch
import cv2
import torchvision.transforms as T
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from coco_colors import get_class_color  # Import the function from coco_colors.py

# Load YOLOv5 model from torch.hub
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' is the small model for faster processing
model_yolo.eval()  # Set the model to evaluation mode

# Load DeepLabV3 model with MobileNetV3 backbone for semantic segmentation
model_deeplab = torch.hub.load('pytorch/vision', 'deeplabv3_mobilenet_v3_large', pretrained=True)
model_deeplab.eval()  # Set the model to evaluation mode

# Initialize webcam capture (0 for default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function for YOLOv5 object detection
def yolo_detection(frame):
    results = model_yolo(frame)  # Detect objects in the current frame
    results.render()  # This adds bounding boxes to the frame
    return frame  # Return the frame with bounding boxes

# Function for DeepLabV3 with MobileNetV3 backbone semantic segmentation
def deeplab_segmentation(frame):
    input_tensor = T.ToTensor()(frame).unsqueeze(0)  # Convert frame to tensor
    with torch.no_grad():
        output = model_deeplab(input_tensor)  # Get segmentation output
        output_predictions = output['out'][0]  # The output tensor for the first image in the batch
        output_predictions = output_predictions.argmax(0)  # Get the class with the maximum score for each pixel

    # Convert segmentation result to NumPy array and resize
    segmentation_mask = Image.fromarray(output_predictions.byte().cpu().numpy())
    segmentation_mask = segmentation_mask.resize((frame.shape[1], frame.shape[0]))  # Resize to match frame size
    segmentation_mask = np.array(segmentation_mask)  # Convert to NumPy array

    # Apply fixed color mapping to segmentation mask
    colored_mask = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3), dtype=np.uint8)
    for class_id in np.unique(segmentation_mask):
        color = get_class_color(class_id)  # Get color from coco_colors.py
        colored_mask[segmentation_mask == class_id] = color

    return colored_mask  # Return the colored segmentation mask

# Use ThreadPoolExecutor to run both tasks in parallel
with ThreadPoolExecutor() as executor:
    while True:
        ret, frame = cap.read()  # Capture each frame from the webcam
        if not ret:
            break

        # Run object detection and segmentation in parallel
        detection_future = executor.submit(yolo_detection, frame)
        segmentation_future = executor.submit(deeplab_segmentation, frame)

        # Get the results (blocks until both tasks are complete)
        frame_with_boxes = detection_future.result()  # Frame with bounding boxes
        segmentation_mask = segmentation_future.result()  # Segmentation mask

        # Ensure the segmentation mask has the same dimensions and 3 channels as the frame
        if segmentation_mask.shape != frame_with_boxes.shape:
            segmentation_mask = cv2.resize(segmentation_mask, (frame_with_boxes.shape[1], frame_with_boxes.shape[0]))

        # Use cv2.addWeighted() for better alpha blending
        # Blending the frame and segmentation mask (adjust the alpha values as needed)
        combined_frame = cv2.addWeighted(frame_with_boxes, 1.0, segmentation_mask, 0.5, 0)

        # Step 5: Show the result
        cv2.imshow("Detection and Segmentation", combined_frame)  # Show frame with bounding boxes and segmentation mask

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
