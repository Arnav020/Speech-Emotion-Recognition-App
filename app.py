# app.py
import streamlit as st
import numpy as np
import os
import librosa
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import librosa.display
import keras
from utils import get_features  

# SETUP 
DATA_DIR = "Data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load model and tools
model = keras.models.load_model("emotion_model.keras")
encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("standard_scaler.pkl")
pca_handcrafted = joblib.load("pca_handcrafted.pkl")

# FUNCTIONS 

def predict_emotion(file_path):
    emotion, avg_probs, labels = get_features(
        file_path,
        pca=pca_handcrafted,
        scaler=scaler,
        get_probs=True,
        model=model,
        encoder=encoder
    )
    return emotion, avg_probs, labels

def plot_spectrogram(file_path):
    y, sr = librosa.load(file_path, duration=2.5, offset=0.6)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set(title='Mel-frequency Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)

def plot_probabilities(probabilities, labels):
    fig, ax = plt.subplots()
    bars = ax.barh(labels, probabilities, color='skyblue')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Emotion Prediction Confidence")
    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{prob:.2f}", va='center')
    st.pyplot(fig)

# UI 

st.set_page_config(page_title="Speech Emotion Recognition 🎙️", layout="centered")
st.title("🎤 Speech Emotion Recognition")
st.write("Upload a `.wav` file to detect the emotion in speech.")

if 'file_path' not in st.session_state:
    st.session_state.file_path = None

# Upload only
uploaded_file = st.file_uploader("📤 Upload Audio File (.wav)", type=["wav"])
if uploaded_file:
    path = os.path.join(DATA_DIR, "uploaded.wav")
    with open(path, "wb") as f:
        f.write(uploaded_file.read())
    st.session_state.file_path = path

# Show Results if File Exists 
if st.session_state.file_path:
    st.audio(st.session_state.file_path, format="audio/wav")

    with st.expander("📊 Show Mel-Spectrogram"):
        plot_spectrogram(st.session_state.file_path)

    if st.button("🚀 Predict Emotion"):
        emotion, probs, labels = predict_emotion(st.session_state.file_path)
        st.success(f"🎯 **Predicted Emotion:** `{emotion}`")
        plot_probabilities(probs, labels)


if __name__ == "__main__":
    pass  # Streamlit runs top-down, so nothing is needed here

