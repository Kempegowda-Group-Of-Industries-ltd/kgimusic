import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model('fer2013_model.h5')
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype('float32') / 255
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    return face_img

def predict_emotion(face_img):
    processed_img = preprocess_image(face_img)
    predictions = model.predict(processed_img)
    max_index = np.argmax(predictions[0])
    emotion = EMOTIONS[max_index]
    return emotion
