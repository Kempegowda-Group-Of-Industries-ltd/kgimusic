import tensorflow as tf
import numpy as np
import cv2

# Load the pre-trained emotion detection model
model = tf.keras.models.load_model('emotion_detection_model.h5')

# List of emotions
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(face_img):
    # Preprocess the image to match the model input
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)

    # Make a prediction
    predictions = model.predict(face_img)
    emotion_index = np.argmax(predictions)
    return EMOTIONS[emotion_index]
