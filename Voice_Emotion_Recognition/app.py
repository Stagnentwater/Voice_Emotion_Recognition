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


with st.spinner("⚙️ Loading the emotion model... (This may take 1–2 minutes on first run)"):
    # Load the pipeine (pre-trained model for emotion detection)
    @st.cache_resource # so that Streamlit caches it and doesn't reload every time
    def load_model():
        return pipeline("audio-classification", model="superb/hubert-large-superb-er")
    emotion_model = load_model()
st.success("✅ Model loaded successfully!")


# If audio is available, make prediction
if wav_audio is not None:
    # --- Predictions ---
    with st.spinner("Analyzing emotion..."):
        try:
            predictions = emotion_model({"array": data, "sampling_rate": sr}, top_k=4)
            all_preds = sorted(predictions, key=lambda x: x['score'], reverse=True)
        except Exception as e:
            st.error(f"Model error: {e}")
            all_preds = []

    if all_preds:
        st.markdown("### 🧠 Detected emotions (all 4 classes)")
        for pred in all_preds:
            bar = int(pred['score'] * 100)
            st.write(f"**{pred['label'].capitalize()}** — {bar:.1f}%")
            st.progress(bar)
        st.success("Analysis complete!")
    else:
        st.warning("No predictions could be made.") 

import matplotlib.pyplot as plt
import librosa.display

# Add visualization of waveform and mel-spectrogram
st.markdown("### 🎵 Visualize your voice")

if wav_audio is not None:
    # 1️⃣ Waveform
    fig, ax = plt.subplots(figsize=(8, 2))
    librosa.display.waveshow(data, sr=sr, ax=ax, color="#2b5876")
    ax.set(title="Waveform", xlabel="Time (s)", ylabel="Amplitude")
    st.pyplot(fig)

    # 2️⃣ Mel-spectrogram
    fig, ax = plt.subplots(figsize=(8, 3))
    mel = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set(title="Mel-Spectrogram")
    st.pyplot(fig)
else:
    st.info("Please record your voice to see visualizations.")
