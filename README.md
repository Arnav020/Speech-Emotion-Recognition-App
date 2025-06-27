# 🎤 Speech Emotion Recognition (SER) App

This project is a **Speech Emotion Recognition (SER)** system that combines **handcrafted audio features** with **Wav2Vec2 embeddings** to classify emotions from `.wav` files. The model is deployed as an interactive **Streamlit web app** where users can upload audio and see emotion predictions along with confidence levels.

---

## 🧠 Model Overview

### 🎯 Objective
To build a model that classifies emotional states (e.g., *angry, calm, happy, fearful, sad, etc.*) from short `.wav` audio clips using both:
- Handcrafted acoustic features
- Deep audio embeddings (Wav2Vec2)

### 📊 Features Used
- **Temporal**: Zero Crossing Rate, RMS Energy  
- **Spectral**: Spectral Centroid, Bandwidth, Rolloff, Contrast  
- **Frequency-based**: MFCCs, Delta MFCCs, Chroma, Mel Spectrogram  
- **Tonal**: Tonnetz, Harmonics  
- **Deep Embedding**: Wav2Vec2 (Facebook's pretrained model)

### ⚙️ Pipeline
1. Load `.wav` file
2. Extract handcrafted features
3. Extract Wav2Vec2 embedding
4. Combine → Apply PCA + Scaling
5. Predict via trained Neural Network
6. Aggregate predictions across augmentations

---

## 🏗️ Model Architecture

- **Input**: `[PCA(handcrafted) + Wav2Vec2 vector]`
- **Layers**:
  - Dense (256), ReLU, Dropout
  - Dense (128), ReLU, Dropout
  - Output Softmax over Emotion Labels
- **Training**: TensorFlow 2.19  
- **Loss**: Categorical Crossentropy  
- **Accuracy**: ~80–85% on validation (depends on dataset used)

---

## 🛠️ Project Structure

```
├── app.py                  # Streamlit frontend  
├── utils.py                # Feature extraction, Wav2Vec2, etc.  
├── emotion_model.keras     # Trained emotion classification model  
├── label_encoder.pkl       # Fitted LabelEncoder  
├── standard_scaler.pkl     # Fitted Scikit-learn scaler  
├── pca_handcrafted.pkl     # PCA model for handcrafted features  
├── requirements.txt  
├── README.md  
└── Data/                   # Uploaded audio files stored here  
```

---

## 🌐 Deployed App on Render

The app is deployed using [Render](https://render.com) for public access and demo.

🔗 **Live App**: [https://speech-emotion-recognition-app.onrender.com](https://speech-emotion-recognition-app.onrender.com)

> ℹ️ *Note: On first load, the Wav2Vec2 model may take a few seconds to download.*

---

## 🚀 Streamlit Web App

### 📌 Features
- Upload `.wav` files  
- Preview audio inline  
- Predict emotion with probabilities  
- Display confidence bar chart  

### 🖥️ How to Run Locally

1. **Clone the Repo**
```bash
git clone https://github.com/yourusername/Speech-Emotion-Recognition-App.git
cd Speech-Emotion-Recognition-App
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the App**
```bash
streamlit run app.py
```

---


## 🧪 Supported Emotions

- Angry  
- Calm  
- Fearful  
- Happy  
- Sad  
- Neutral  
- Disgust  
- Surprise
