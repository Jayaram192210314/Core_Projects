import cv2

class FaceDetector:
    def __init__(self, faceCascadePath):
        self.faceCascade = cv2.CascadeClassifier(faceCascadePath)

    def detect(self, image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        rects = self.faceCascade.detectMultiScale(
            image,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=minSize,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return rects

# Specify file paths
face_cascade_path = "haarcascade_frontalface_default.xml"
image_path = "face.jpg"

# Load the input image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image file not found at {image_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the face detector and detect faces
fd = FaceDetector(face_cascade_path)
faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
print(f"I found {len(faceRects)} face(s)")

# Draw rectangles around detected faces
for (x, y, w, h) in faceRects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the output
cv2.imshow("Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
