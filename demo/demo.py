import streamlit as st
import tempfile
import os
import time
import numpy as np
import torchaudio
import torch
from faster_whisper import WhisperModel
from audio_recorder_streamlit import audio_recorder

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "fine_tune_model")

@st.cache_resource
def load_faster_whisper_model(model_dir):
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "float32"

    model = WhisperModel(model_dir, device=device, compute_type=compute_type)
    return model, device

def load_audio(file_bytes: bytes, file_name: str = "audio.wav", target_sr: int = 16000):
    """Load audio using torchaudio and convert to mono"""
    ext = os.path.splitext(file_name)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_name = tmp.name
    finally:
        tmp.close()

    waveform, sr = torchaudio.load(tmp_name)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
        sr = target_sr

    audio_np = waveform.cpu().numpy().astype(np.float32)

    try:
        os.remove(tmp_name)
    except Exception:
        pass

    return audio_np, sr

def transcribe_audio_faster_whisper(model, audio_np: np.ndarray, sr: int):
    """Transcribe full audio using Faster Whisper and return concatenated text"""
    segments, info = model.transcribe(audio_np, beam_size=5, vad_filter=True)
    text = " ".join([seg.text for seg in segments])
    return text

def main():
    st.set_page_config(page_title="PhoWhisper Faster Whisper Demo", layout="centered")
    st.title("🎙️ PhoWhisper Faster Whisper Demo")
    st.write(
        "Upload an audio file (wav, mp3, m4a, etc.) or record audio, then click Transcribe. "
    )

    # Load model
    with st.spinner("Loading Faster Whisper model..."):
        model, device = load_faster_whisper_model(MODEL_DIR)
    st.success(f"Model loaded on {device}")
    st.markdown("---")

    mode = st.selectbox(
        "Choose input method:",
        options=["Upload file", "Record with microphone"],
        index=0,
    )
    
    uploaded_bytes = None
    file_name = None

    if mode == "Upload file":
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload audio file", type=["wav", "mp3", "m4a", "flac", "ogg"], accept_multiple_files=False
        )

        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.read()
            file_name = uploaded_file.name
            st.audio(uploaded_bytes)
    else:
        audio_bytes = audio_recorder(
            text="Record / Stop", 
            recording_color="#e74c3c", 
            neutral_color="#2ecc71", 
            icon_name="microphone", 
            sample_rate=16000,
            icon_size="1x"
        )
        st.info("Press the button below to start recording (press again to stop):")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            uploaded_bytes = audio_bytes
            file_name = "recorded_audio.wav"

    if st.button("Transcribe"):
        if uploaded_bytes is None:
            st.warning("Please upload an audio file before transcribing.")
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

            st.success("Transcription complete")
            st.write(f"_Elapsed: {duration:.2f}s_")
            st.text_area("Transcribed text", value=text, height=150, disabled=True)

if __name__ == "__main__":
    main()