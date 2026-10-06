"""Audio feature extraction: raw waveform -> log-mel spectrogram.

A neural network cannot read raw audio directly in a useful way, so we
convert each 1-second clip into a 2D "image" called a log-mel spectrogram:
rows = pitch (mel-scaled frequency), columns = time. See FIRST_PRINCIPLES.md
for *why* this representation works.
"""

import soundfile as sf
import torch
import torchaudio

SAMPLE_RATE = 16000   # Google Speech Commands clips are 16 kHz mono
N_MELS = 40           # number of mel filterbank channels (rows of the image)
N_FFT = 512           # FFT window size
WIN_LENGTH = 400      # 25 ms analysis window  (400 samples @ 16 kHz)
HOP_LENGTH = 160      # 10 ms hop between windows (160 samples @ 16 kHz)
CLIP_SAMPLES = 16000  # every clip is exactly 1 second

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0,
)
_db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)


def fix_length(waveform: torch.Tensor, target: int = CLIP_SAMPLES) -> torch.Tensor:
    """Pad (with zeros) or trim a 1-D waveform to exactly `target` samples."""
    if waveform.numel() > target:
        return waveform[:target]
    if waveform.numel() < target:
        pad = torch.zeros(target - waveform.numel(), dtype=waveform.dtype)
        return torch.cat([waveform, pad])
    return waveform


def waveform_to_logmel(waveform: torch.Tensor) -> torch.Tensor:
    """Convert a mono waveform of shape (16000,) to a log-mel spectrogram.

    Returns a tensor of shape (1, 40, 101): 1 channel, 40 mel bins, ~101
    time frames. Values are log-scaled and normalized to zero mean/unit
    variance *per clip*, which makes training much more stable.
    """
    waveform = fix_length(waveform).unsqueeze(0)          # (1, 16000)
    mel = _mel_transform(waveform)                        # (1, 40, 101) power spectrogram
    logmel = _db_transform(mel)                           # log scale -> decibels
    mean = logmel.mean()
    std = logmel.std().clamp(min=1e-6)
    return (logmel - mean) / std                           # per-utterance normalization


def load_waveform(path: str) -> torch.Tensor:
    """Load any wav file as a mono 16 kHz float tensor of shape (16000,)."""
    wav_np, sr = sf.read(path, dtype="float32")  # (samples[, channels])
    wav = torch.from_numpy(wav_np)
    if wav.dim() > 1:                            # stereo -> mono
        wav = wav.mean(dim=1)
    if sr != SAMPLE_RATE:                        # resample if needed
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return fix_length(wav)
