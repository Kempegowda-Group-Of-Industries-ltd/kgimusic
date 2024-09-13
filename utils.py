import cv2

# Function to detect and crop face using OpenCV's Haarcascade model
def crop_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    
    if len(faces) == 0:
        return None
    
    (x, y, w, h) = faces[0]
    cropped_face = gray_frame[y:y+h, x:x+w]
    return cropped_face

# Recommend songs based on the detected emotion
def get_recommendations(emotion_list):
    song_dict = {
        'Happy': ['Happy Song 1', 'Happy Song 2', 'Happy Song 3'],
        'Sad': ['Sad Song 1', 'Sad Song 2', 'Sad Song 3'],
        'Angry': ['Angry Song 1', 'Angry Song 2', 'Angry Song 3'],
        'Surprise': ['Surprise Song 1', 'Surprise Song 2', 'Surprise Song 3'],
        'Neutral': ['Neutral Song 1', 'Neutral Song 2', 'Neutral Song 3'],
        'Fear': ['Fear Song 1', 'Fear Song 2', 'Fear Song 3'],
        'Disgust': ['Disgust Song 1', 'Disgust Song 2', 'Disgust Song 3']
    }
    
    recommendations = []
    for emotion in emotion_list:
        recommendations += song_dict.get(emotion, ['Default Song'])
    return recommendations
