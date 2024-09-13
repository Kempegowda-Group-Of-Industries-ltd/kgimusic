import mediapipe as mp
import cv2

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Create Face Detection object
face_detection = mp_face_detection.FaceDetection()

# Read image using OpenCV
image = cv2.imread('your_image.jpg')  # Replace 'your_image.jpg' with your image file path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces
results = face_detection.process(image_rgb)

if results.detections:
    for detection in results.detections:
        mp_drawing.draw_detection(image, detection)

# Display the image with detected faces
cv2.imshow('MediaPipe Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
