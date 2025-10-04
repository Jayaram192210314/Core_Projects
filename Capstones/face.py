# Import the OpenCV library
import cv2

# Initialize the face cascade using the frontal face Haar cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Desired output dimensions
OUTPUT_SIZE_WIDTH = 700
OUTPUT_SIZE_HEIGHT = 600

# Open the first webcam device
capture = cv2.VideoCapture(0)

# Create OpenCV named windows for showing the input and output images
cv2.namedWindow("base-image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("result-image", cv2.WINDOW_AUTOSIZE)

# Position the windows
cv2.moveWindow("base-image", 20, 200)
cv2.moveWindow("result-image", 640, 200)

# Start the OpenCV window thread
cv2.startWindowThread()

# Rectangle color in BGR
rectangleColor = (0, 100, 255)

# Loop for real-time face detection
while True:
    # Capture a frame from the webcam
    ret, frame = capture.read()
    if not ret:
        print("Error capturing frame from webcam.")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Copy the frame for drawing rectangles
    result_frame = frame.copy()

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), rectangleColor, 2)

    # Resize the frames to the desired output dimensions
    base_image_resized = cv2.resize(frame, (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))
    result_image_resized = cv2.resize(result_frame, (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))

    # Display the frames
    cv2.imshow("base-image", base_image_resized)
    cv2.imshow("result-image", result_image_resized)

    # Exit the loop if 'q' or 'Esc' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

# Release the webcam and close all windows
capture.release()
cv2.destroyAllWindows()
