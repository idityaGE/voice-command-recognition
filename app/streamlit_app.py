"""Live demo: speak (or upload) a 1-second clip, get the predicted word.

Run:
    streamlit run app/streamlit_app.py

Uses st.audio_input (browser mic recording) with a file-upload fallback.
Loads models/keyword_cnn.pt by default; use --custom to load the fine-tuned
models/keyword_cnn_custom.pt instead (sidebar checkbox does this too).
"""

import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import waveform_to_logmel, SAMPLE_RATE
from models import KeywordCNN

ROOT = Path(__file__).resolve().parent.parent


@st.cache_resource
def load_model(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = KeywordCNN(num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["classes"]


def predict(wav_bytes: bytes, model, classes):
    import soundfile as sf
    wav, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        import torchaudio
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), sr, SAMPLE_RATE).numpy()
    # center-crop / pad to exactly 1 second
    target = SAMPLE_RATE
    if len(wav) > target:
        start = (len(wav) - target) // 2
        wav = wav[start:start + target]
    else:
        wav = np.pad(wav, (0, target - len(wav)))
    logmel = waveform_to_logmel(torch.from_numpy(wav)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(logmel), dim=1).squeeze(0).numpy()
    top3 = sorted(zip(classes, probs), key=lambda t: -t[1])[:3]
    return top3


st.set_page_config(page_title="Voice Command Recognition", page_icon="🎙️")
st.title("🎙️ Voice Command Recognition")
st.write("Speak a command (or upload a 1-second `.wav` clip) — the CNN "
         "predicts which of the trained words it heard.")

use_custom = st.sidebar.checkbox(
    "Use my fine-tuned model (keyword_cnn_custom.pt)",
    value=(ROOT / "models" / "keyword_cnn_custom.pt").exists())
ckpt_path = (ROOT / "models" /
             ("keyword_cnn_custom.pt" if use_custom else "keyword_cnn.pt"))
if not ckpt_path.exists():
    st.error(f"Weights not found: {ckpt_path}\n"
             "Train first: `python src/train.py --data data`")
    st.stop()

model, classes = load_model(str(ckpt_path))
st.sidebar.write(f"Loaded **{len(classes)}** word classes.")

audio = None
try:
    # browser mic recording (streamlit >= 1.35)
    audio = st.audio_input("Record a 1-second command")
except Exception:
    st.info("Mic recording not supported by this Streamlit version — "
            "use the file upload below.")
if audio is None:
    uploaded = st.file_uploader("…or upload a .wav clip", type=["wav"])
    audio = uploaded

if audio is not None:
    data = audio.read() if hasattr(audio, "read") else audio.getvalue()
    top3 = predict(data, model, classes)
    st.subheader(f'Heard: "{top3[0][0]}"  ({top3[0][1]:.1%} confident)')
    for word, p in top3:
        st.write(f"{word}")
        st.progress(float(p))
    st.caption("Tip: the model expects ~1 second of audio; longer clips are "
               "center-cropped.")
