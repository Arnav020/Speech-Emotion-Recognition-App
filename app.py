# app.py
import streamlit as st
import numpy as np
import sounddevice as sd
import os
from scipy.io.wavfile import write
import librosa
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import librosa.display
from utils import get_features  

# SETUP 
DATA_DIR = "Data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load model and tools
model = tf.keras.models.load_model("emotion_model.keras")
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

def save_recording(audio, sample_rate=22050):
    existing = [int(f.split(".")[0]) for f in os.listdir(DATA_DIR) if f.endswith(".wav") and f.split(".")[0].isdigit()]
    next_id = max(existing, default=0) + 1
    filename = f"{next_id:04d}.wav"
    path = os.path.join(DATA_DIR, filename)
    write(path, sample_rate, audio)
    return path

def record_audio(duration=3, sample_rate=22050):
    st.info("🔴 Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    st.success("✅ Recording complete!")
    return save_recording(audio, sample_rate)

# UI 

st.set_page_config(page_title="Speech Emotion Recognition 🎙️", layout="centered")
st.title("🎤 Speech Emotion Recognition")
st.write("Upload a `.wav` file or record your voice to detect the emotion.")

if 'file_path' not in st.session_state:
    st.session_state.file_path = None

# Input Options 
option = st.radio("Choose input method:", ["Upload .wav file", "Record using microphone"])

if option == "Upload .wav file":
    uploaded_file = st.file_uploader("Upload Audio", type=["wav"])
    if uploaded_file:
        path = os.path.join(DATA_DIR, "uploaded.wav")
        with open(path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.file_path = path

elif option == "Record using microphone":
    if st.button("🎙️ Record Now"):
        st.session_state.file_path = record_audio()

# Show Results if File Exists 
if st.session_state.file_path:
    st.audio(st.session_state.file_path, format="audio/wav")

    with st.expander("📊 Show Mel-Spectrogram"):
        plot_spectrogram(st.session_state.file_path)

    if st.button("🚀 Predict Emotion"):
        emotion, probs, labels = predict_emotion(st.session_state.file_path)
        st.success(f"🎯 **Predicted Emotion:** `{emotion}`")
        plot_probabilities(probs, labels)
