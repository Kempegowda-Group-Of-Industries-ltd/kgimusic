import streamlit as st
import cv2
import numpy as np
from PIL import Image
from model import predict_emotion
from utils import crop_face, get_recommendations

# Streamlit app title
st.title("Emotion-Based Music Recommendation System")

# Start the camera or upload an image
st.subheader("Choose an input method:")

input_method = st.radio("Select input method", ("Use Webcam", "Upload Image"))

# Handling Webcam Input
if input_method == "Use Webcam":
    run = st.checkbox('Start Camera')
    if run:
        cap = cv2.VideoCapture(0)
        FRAME_WINDOW = st.image([])

        while run:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            cropped_face = crop_face(frame)
            
            if cropped_face is not None:
                emotion = predict_emotion(cropped_face)
                st.subheader(f"Detected Emotion: {emotion}")
                recommendations = get_recommendations([emotion])
                st.expander("Click to see recommendations").write(recommendations)
else:
    # Handling Image Upload
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Processing...")
        
        frame = np.array(image)
        cropped_face = crop_face(frame)
        
        if cropped_face is not None:
            emotion = predict_emotion(cropped_face)
            st.subheader(f"Detected Emotion: {emotion}")
            recommendations = get_recommendations([emotion])
            st.expander("Click to see recommendations").write(recommendations)

# CSS Styling for UI enhancement
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
