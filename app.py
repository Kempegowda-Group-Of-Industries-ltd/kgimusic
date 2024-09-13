import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import face_recognition
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

# Load your pre-trained emotion recognition model
MODEL_PATH = 'fer2013_model.h5'
if not os.path.isfile(MODEL_PATH):
    st.error("Model file not found.")
else:
    model = load_model(MODEL_PATH)

# Spotify API setup
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='your_client_id',
                                                           client_secret='your_client_secret'))

def get_recommendations(emotion):
    # Define emotion to playlist mapping
    playlists = {
        'happy': '37i9dQZF1DWU2tF5wYF0b7',
        'sad': '37i9dQZF1DWTk6J3Jc2V9b',
        'angry': '37i9dQZF1DWU2BSc7nv0sb',
        'surprised': '37i9dQZF1DWTbA7b4FZxKt'
    }
    playlist_id = playlists.get(emotion, '37i9dQZF1DWU2tF5wYF0b7')  # Default to happy playlist
    results = sp.playlist_tracks(playlist_id)
    tracks = [track['track']['name'] for track in results['items']]
    return tracks

def main():
    st.title("Emotion-based Music Recommendation")

    # Start Camera button
    if st.button("Start Camera"):
        st.write("Starting camera...")
        # OpenCV camera capture
        cap = cv2.VideoCapture(0)

        stframe = st.empty()
        
        # Capture video from the camera
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame.")
                break
            
            # Process the frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_image = rgb_frame[top:bottom, left:right]
                face_image_resized = cv2.resize(face_image, (48, 48))
                face_image_array = np.array(face_image_resized) / 255.0
                face_image_array = np.expand_dims(face_image_array, axis=0)
                face_image_array = np.expand_dims(face_image_array, axis=-1)
                
                # Predict emotion
                emotion_prediction = model.predict(face_image_array)
                emotion = np.argmax(emotion_prediction)
                
                # Map emotion index to emotion label
                emotions = ['happy', 'sad', 'angry', 'surprised']
                detected_emotion = emotions[emotion]
                
                # Display detected emotion
                st.write(f"Detected Emotion: {detected_emotion}")
                
                # Get recommendations based on emotion
                recommendations = get_recommendations(detected_emotion)
                st.write(f"Recommendations for {detected_emotion}:")
                st.write(recommendations)
                
            stframe.image(frame, channels='BGR', use_column_width=True)
            
            # Stop camera if user presses stop button
            if st.button('Stop Camera'):
                cap.release()
                st.write("Camera stopped.")
                break

if __name__ == "__main__":
    main()
