import tensorflow as tf
import numpy as np
import cv2

# Load pre-trained FER2013 model (make sure to have 'fer2013_model.h5' in the same directory)
model = tf.keras.models.load_model('fer2013_model.h5')

# Emotion labels from FER2013
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(face_img):
    """Preprocess the face image to match the model input."""
    face_img = cv2.resize(face_img, (48, 48))  # Resize image to 48x48 pixels
    face_img = face_img.astype('float32') / 255  # Normalize pixel values
    face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
    face_img = np.expand_dims(face_img, axis=-1)  # Add channel dimension for grayscale
    return face_img

def predict_emotion(face_img):
    """Predict emotion from a cropped face image."""
    processed_img = preprocess_image(face_img)
    predictions = model.predict(processed_img)
    max_index = np.argmax(predictions[0])
    emotion = EMOTIONS[max_index]
    return emotion
