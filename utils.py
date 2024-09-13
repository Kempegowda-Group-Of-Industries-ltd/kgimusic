from PIL import Image
import face_recognition
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np

# Face Cropping Function
def crop_face(frame):
    # Convert the frame to a NumPy array
    frame_array = np.array(frame)
    
    # Detect faces
    face_locations = face_recognition.face_locations(frame_array)
    
    if len(face_locations) == 0:
        return None
    
    # Use the first detected face
    top, right, bottom, left = face_locations[0]
    cropped_face = frame.crop((left, top, right, bottom))
    
    return cropped_face

# Spotify API setup
client_credentials_manager = SpotifyClientCredentials(client_id='your_spotify_client_id', client_secret='your_spotify_client_secret')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Recommend Songs Based on Detected Emotions
def get_recommendations(emotion_list):
    emotion_song_map = {
        'Happy': 'happy',
        'Sad': 'sad',
        'Angry': 'angry',
        'Surprise': 'surprise',
        'Neutral': 'chill',
        'Fear': 'calm',
        'Disgust': 'energetic'
    }
    
    recommendations = []
    
    for emotion in emotion_list:
        if emotion in emotion_song_map:
            search_query = emotion_song_map[emotion] + " music"
            results = sp.search(q=search_query, type='track', limit=5)
            for track in results['tracks']['items']:
                recommendations.append(track['name'] + ' by ' + track['artists'][0]['name'])

    return recommendations
