import cv2
import streamlit as st
import numpy as np
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

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

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ----------------------------
# WebRTC Video Transformer
# ----------------------------
class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]

            # Resize to model input
            resized = cv2.resize(face_img, (48, 48))
            rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img_array = np.expand_dims(rgb_img / 255.0, axis=0)

            # Predict
            prediction = model.predict(img_array, verbose=0)[0]
            emotion = emotion_labels[np.argmax(prediction)]
            color = emotion_colors[emotion]

            # Draw face box
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)

            # Display emotion text
            cv2.putText(img, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return img

# ----------------------------
# WebRTC Stream
# ----------------------------
webrtc_streamer(
    key="emotion",
    video_transformer_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False}
)

