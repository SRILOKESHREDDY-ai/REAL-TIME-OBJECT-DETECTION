import cv2

# Initialize webcam capture (0 for default webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Capture video feed from the webcam
while True:
    ret, frame = cap.read()  # Read the frame from the webcam
    if not ret:
        break

    # Show the frame in a window
    cv2.imshow("Webcam Feed", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
