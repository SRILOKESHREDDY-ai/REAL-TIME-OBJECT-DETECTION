import torch
import torchvision.models.segmentation
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image

# Load DeepLabV3 with MobileNetV3 as the backbone
model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Initialize webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to process segmentation
def deeplab_mobilenetv3_segmentation(frame):
    input_tensor = T.ToTensor()(frame).unsqueeze(0)  # Convert frame to tensor
    input_tensor = input_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move to GPU if available
    with torch.no_grad():
        output = model(input_tensor)  # Get segmentation output
        output_predictions = output['out'][0]  # Output tensor
        output_predictions = output_predictions.argmax(0)  # Get the class with the highest score for each pixel
    return output_predictions

# Use the model to process frames
while True:
    ret, frame = cap.read()  # Read the frame from the webcam
    if not ret:
        break

    # Perform segmentation using DeepLabV3 with MobileNetV3
    segmentation_mask = deeplab_mobilenetv3_segmentation(frame)

    # Convert segmentation result to NumPy array and resize to match frame size
    segmentation_mask = Image.fromarray(segmentation_mask.byte().cpu().numpy())
    segmentation_mask = segmentation_mask.resize((frame.shape[1], frame.shape[0]))
    segmentation_mask = np.array(segmentation_mask)

    # Convert the segmentation mask to 3 channels (for overlay)
    segmentation_mask = np.stack([segmentation_mask] * 3, axis=-1)

    # Overlay the segmentation mask on the frame (using alpha blending)
    combined_frame = cv2.addWeighted(frame, 1, segmentation_mask, 0.5, 0)

    # Show the frame with segmentation mask
    cv2.imshow("DeepLabV3 with MobileNetV3 Segmentation", combined_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
