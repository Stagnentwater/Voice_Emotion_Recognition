# 🎙️ Voicy — Voice Emotion Detection App

**Voicy** is an AI-powered Streamlit web app that detects emotions from your voice using a pretrained model.  
It allows you to record your voice directly in the browser, visualize the waveform and spectrogram,  
and uses a pre-trained **speech emotion recognition model** from Hugging Face to classify your emotion in real time.  

---

## 🚀 Live Demo
👉 [Try it on Streamlit Cloud](https://voicy--nafisat-ibrahim.streamlit.app) : https://voicy--nafisat-ibrahim.streamlit.app/

## 🎬 Demo Video

Here’s a quick demo of **Voicy: Voice Emotion Detection** in action 🎙️👇  

<div align="center">

  <video 
    src="https://private-user-images.githubusercontent.com/135756896/500352996-a315aced-93da-4a21-acc0-e278c27ec58f.mp4"
    controls
    muted
    loop
    width="700"
    style="border-radius:12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
    Your browser does not support the video tag.
  </video>

  <br><br>

  <em>🎞️ Scene: “We were too late.” — Dory & Marlin, <strong>Finding Nemo (2003)</strong>  
  Detected emotion: <strong>Sad (63%)</strong></em>

</div>



---

## 🧠 Overview

Voicy was built as a learning project to explore **audio signal processing** and **speech emotion recognition (SER)**.  
It combines simple Streamlit UI components with a Hugging Face transformer model to analyze recorded speech and infer emotional tone.  

When you speak, your browser records the audio and the app:
1. Captures your voice as a `.wav` file (via `st_audiorec`)
2. Converts it into a NumPy waveform
3. Resamples it to 16 kHz mono (the standard input format for most speech models)
4. Visualizes the waveform and Mel-spectrogram
5. Runs the pre-trained model [`superb/hubert-large-superb-er`](https://huggingface.co/superb/hubert-large-superb-er)
6. Displays the top predicted emotions with their confidence scores

---

## 🧩 Model Details

The app uses the **HuBERT (Hidden-Unit BERT)** model fine-tuned for speech emotion recognition:

- **Model name:** [`superb/hubert-large-superb-er`](https://huggingface.co/superb/hubert-large-superb-er)
- **Architecture:** HuBERT Large (self-supervised speech model)
- **Task:** Emotion Recognition (SER)
- **Input:** Raw speech (16 kHz, mono)
- **Output:** Predicted emotion categories such as *Happy*, *Sad*, *Angry*, *Neutral*, *Disgust*, etc.
- **Source:** [Hugging Face – SUPERB Benchmark](https://huggingface.co/superb)

HuBERT learns hidden units from speech without needing manual labels and achieves state-of-the-art performance across multiple speech-related tasks.

---

## 🧰 Tech Stack

| Component | Description |
|------------|-------------|
| **Streamlit** | Web framework for interactive apps |
| **st_audiorec** | Streamlit component for audio recording |
| **Librosa** | Audio processing (resampling, visualization) |
| **Matplotlib** | Plotting waveform and spectrogram |
| **Transformers** | Hugging Face model pipeline |
| **Torch** | Backend for model inference |

<P>🤖 Model powered by <a href="https://huggingface.co/superb/hubert-large-superb-er"><strong>HuBERT (superb/hubert-large-superb-er)</strong></a> — a speech emotion recognition model by <a href="https://huggingface.co/superb">Hugging Face SUPERB</a>.</sub>

---

## 📦 Installation

Clone this repository:

```bash
git clone https://github.com/Nafisatibrahim/Voice_Emotion_Recognition.git
cd Voice_Emotion_Recognition
