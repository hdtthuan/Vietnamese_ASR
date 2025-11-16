import streamlit as st
import tempfile
import os
import time
import numpy as np
import torch
import librosa

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_BASE_DIR, ".."))
MODEL_DIR = os.path.join(_PROJECT_ROOT, "fine_tune_model")

_CACHE_DIR = os.path.join(_PROJECT_ROOT, ".cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = _CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = _CACHE_DIR

from faster_whisper import WhisperModel
from audio_recorder_streamlit import audio_recorder


@st.cache_resource
def load_faster_whisper_model(model_dir):
    if not os.path.isdir(model_dir):
        st.error(f"Model directory not found at: {model_dir}")
        st.error("Please make sure you have run the convert_model.py script and that the MODEL_DIR path is correct.")
        return None, None

    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "float32"
    
    model = WhisperModel(model_dir, device=device, compute_type=compute_type)
    return model, device

def load_audio(file_bytes: bytes, file_name: str = "audio.wav", target_sr: int = 16000):
    ext = os.path.splitext(file_name)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_name = tmp.name
    finally:
        tmp.close()

    try:
        audio_np, sr = librosa.load(tmp_name, sr=target_sr, mono=True)
    finally:
        try:
            os.remove(tmp_name)
        except Exception:
            pass

    return audio_np, sr 

def transcribe_audio_faster_whisper(model, audio_np: np.ndarray, sr: int):
    segments, info = model.transcribe(audio_np, beam_size=5, vad_filter=True)
    text = " ".join([seg.text for seg in segments])
    return text

def main():
    st.set_page_config(page_title="PhoWhisper Faster Whisper Demo", layout="centered")
    st.title("PhoWhisper Faster Whisper Demo")
    st.write(
        "Upload an audio file (wav, mp3, m4a...) or record audio, then press Transcribe."
    )

    with st.spinner("Loading Faster Whisper model..."):
        model, device = load_faster_whisper_model(MODEL_DIR)
        if model is None:
            return

    st.success(f"Model loaded successfully on {device.upper()}")
    st.markdown("---")

    # Chọn chế độ: Upload hoặc Ghi âm
    input_method = st.selectbox(
        "Select input method:",
        options=["Upload file", "Record with microphone"],
        index=0,
    )
    
    uploaded_bytes = None
    file_name = None

    if input_method == "Upload file":
        uploaded_file = st.file_uploader(
            "Upload an audio file", type=["wav", "mp3", "m4a", "flac", "ogg"], accept_multiple_files=False
        )

        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.read()
            file_name = uploaded_file.name
            st.audio(uploaded_bytes)
            
    else:
        st.info("Press the button below to start recording (press again to stop):")
        audio_bytes = audio_recorder(
            text="Record / Stop", 
            recording_color="#e74c3c", 
            neutral_color="#2ecc71", 
            sample_rate=16000,
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            uploaded_bytes = audio_bytes
            file_name = "recorded_audio.wav"

    if st.button("Transcribe"):
        if uploaded_bytes is None:
            st.warning("Please upload or record an audio file first.")
        else:
            with st.spinner("Transcribing..."):
                start = time.time()
                try:
                    audio_np, sr = load_audio(uploaded_bytes, file_name, target_sr=16000)
                    text = transcribe_audio_faster_whisper(model, audio_np, sr)
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    return
                
                duration = time.time() - start

            st.success("Done!")
            st.write(f"_Processing time: {duration:.2f} seconds_")
            st.text_area("Transcribed Text", value=text, height=150, disabled=True)

if __name__ == "__main__":
    main()
