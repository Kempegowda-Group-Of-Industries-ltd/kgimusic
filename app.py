import streamlit as st
from PIL import Image
import cv2
import numpy as np
from utils import crop_face, get_recommendations
from model import predict_emotion

# Streamlit app setup
st.set_page_config(page_title="Emotion-Based Music Recommender", page_icon="🎶", layout="wide")
st.title("🎶 Emotion-Based Music Recommendation System")

st.markdown("""
<style>
    .stButton>button {
        background-color: #FF6347;
        color: white;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF4500;
    }
</style>
""", unsafe_allow_html=True)

# Placeholder for camera feed
run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)
emotion_list = []

if run:
    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect and crop the face
        frame_img = Image.fromarray(frame)
        cropped_face = crop_face(frame_img)
        
        if cropped_face:
            # Predict emotion for the cropped face
            emotion = predict_emotion(np.array(cropped_face))
            emotion_list.append(emotion)
        
        FRAME_WINDOW.image(frame)
        
        if len(emotion_list) >= 40:
            break

    st.success("✔️ Detected Emotions")
    st.write("Emotions: ", emotion_list)

    # Get unique emotions
    unique_emotions = sorted(set(emotion_list), key=lambda x: emotion_list.count(x), reverse=True)
    st.write("🎭 Final Emotions: ", unique_emotions)

    # Recommend songs based on emotions
    recommendations = get_recommendations(unique_emotions)
    st.subheader("🎶 Recommended Songs")
    for song in recommendations:
        st.write(f"- {song}")

else:
    st.write("Camera stopped.")



st.markdown("""
<style>
    .stButton>button {
        background-color: #FF6347;
        color: white;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF4500;
    }
    .expander-content {
        animation: slideDown 0.5s ease-out;
    }
    @keyframes slideDown {
        from {
            max-height: 0;
            opacity: 0;
        }
        to {
            max-height: 100%;
            opacity: 1;
        }
    }
</style>
""", unsafe_allow_html=True)

st.expander("Click to see recommendations", expanded=True).write("Recommendations will appear here after detection.")

