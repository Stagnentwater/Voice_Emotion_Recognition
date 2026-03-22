# Purpose: Detect emotions based on voice input using a pre-trained model.

from st_audiorec import st_audiorec
import streamlit as st
import io, soundfile as sf
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from transformers import pipeline

# Set the title of the app
st.title("Voicy: Voice Emotion Detection")

# Brief description of what the app does
st.write("Voicy is an AI-powered voice emotion detection. It takes a speech recording as an input and classifies the emotion of the person talking. ")

# Get audio from user
st.subheader("🎤 Record your voice")
wav_audio = st_audiorec() # to show UI button

if wav_audio is not None:
    # Playback right away
    st.audio(wav_audio, format="audio/wav")

    # Convert WAV bytes to numpy waveform + sample rate
    data, sr = sf.read(io.BytesIO(wav_audio), dtype="float32", always_2d=False)

    # If stereo, average to mono now or later:
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Convert sample rate to 16,000 Hz
    target = 16000
    if sr != target:
        data = librosa.resample(data, orig_sr = sr, target_sr=target)
        sr = target

    st.success(f"Captured {len(data)/sr:.2f}s at {sr} Hz")

    
# Load the pipeine (pre-trained model for emotion detection)
@st.cache_resource # so that Streamlit caches it and doesn't reload every time

def load_model():
    return pipeline("audio-classification", model="superb/hubert-large-superb-er")

emotion_model = load_model()

# If audio is available, make prediction
if wav_audio is not None:
    with st.spinner("Analyzing emotion..."):
        # Make prediction
        predictions = emotion_model({"array": data, "sampling_rate": sr}, top_k=3)
        st.success("Done!")
        
        # Display the results
        st.subheader("🗂️ Top Predictions")
        for pred in predictions:
            st.write(f"**{pred['label']}** with confidence {pred['score']:.2f}")
        
        # Plot the waveform
        st.subheader("📈 Waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(data, sr=sr, ax=ax)
        ax.set_title("Waveform")
        st.pyplot(fig)