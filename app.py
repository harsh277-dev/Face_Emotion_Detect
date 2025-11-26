import cv2
import streamlit as st
import numpy as np
from keras.models import load_model

# ------------------------------
# Load model safely
# ------------------------------
@st.cache_resource
def load_emotion_model():
    return load_model("emotion_model_v4.h5")

model = load_emotion_model()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emotion colors
emotion_colors = {
    "Angry": (0, 0, 255),
    "Disgust": (0, 255, 0),
    "Fear": (255, 0, 0),
    "Happy": (0, 255, 255),
    "Neutral": (255, 255, 255),
    "Sad": (255, 0, 255),
    "Surprise": (0, 128, 255),
}

st.set_page_config(page_title="Advanced Emotion Detector", layout="wide")
st.title("🎭 Advanced Emotion Detection System")

# Camera input (browser-based)
img_file = st.camera_input("Take a picture")

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if img_file is not None:
    # Convert uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # Resize to model input
        resized = cv2.resize(face_img, (48, 48))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_array = np.expand_dims(rgb_img / 255.0, axis=0)

        # Predict
        prediction = model.predict(img_array, verbose=0)[0]
        emotion = emotion_labels[np.argmax(prediction)]
        color = emotion_colors[emotion]

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

        # Display emotion text
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show processed frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame, caption="Detected Emotion(s)")
