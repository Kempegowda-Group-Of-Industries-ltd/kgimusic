import streamlit as st
import cv2
import numpy as np
from model import predict_emotion
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

# Setup Spotify client
sp = Spotify(auth_manager=SpotifyClientCredentials(client_id='YOUR_SPOTIPY_CLIENT_ID',
                                                   client_secret='YOUR_SPOTIPY_CLIENT_SECRET'))

# Define emotion-to-song mapping
emotion_to_song = {
    0: 'happy',    # Replace with actual emotion mappings
    1: 'sad',
    2: 'angry',
    3: 'neutral',
    # Add other emotions
}

def get_recommendations(emotion):
    genre = emotion_to_song.get(emotion, 'happy')
    results = sp.search(q=f'genre:{genre}', type='track', limit=5)
    tracks = results['tracks']['items']
    return [(track['name'], track['artists'][0]['name']) for track in tracks]

st.title('Emotion-Based Music Recommendation')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    emotion = predict_emotion(image)
    
    if emotion is not None:
        st.image(image, channels="BGR")
        st.write(f"Predicted emotion: {emotion_to_song.get(emotion, 'Unknown')}")
        
        recommendations = get_recommendations(emotion)
        st.write("Recommended songs:")
        for track in recommendations:
            st.write(f"{track[0]} by {track[1]}")
    else:
        st.write("No face detected or unable to predict emotion.")
