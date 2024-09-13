import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Load the trained emotion model
MODEL_PATH = 'emotion_model_sample.h5'

model = tf.keras.models.load_model(MODEL_PATH)

def detect_face(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            return bbox
    return None

def preprocess_image(image, bbox):
    x, y, w, h = bbox
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))  # assuming the model expects 48x48 images
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face

def predict_emotion(image):
    bbox = detect_face(image)
    if bbox:
        face = preprocess_image(image, bbox)
        prediction = model.predict(face)
        return np.argmax(prediction)
    return None
