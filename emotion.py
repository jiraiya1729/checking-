import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

# Get the path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the emotion model file in the 'inbuilt' folder
emotion_model_path = os.path.join(current_dir, "../inbuilt/emotion_model.h5")
emotion_model = load_model(emotion_model_path)

# Define the emotions labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion(face_img):
    # Preprocess the image
    face_img = cv2.resize(face_img, (48, 48))  # Resize to match model input size
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension

    # Perform emotion prediction
    emotion_probs = emotion_model.predict(face_img)
    emotion_label = emotions[np.argmax(emotion_probs)]

    return emotion_label

# Example usage
if __name__ == "__main__":
    # Load a sample face image
    face_img = cv2.imread('path/to/your/face_image.jpg')

    # Detect emotion
    emotion_label = detect_emotion(face_img)

    print("Emotion:", emotion_label)
