"""On-the-fly augmentation for keyword-spotting training.

All functions work on raw waveforms or log-mel tensors so they can be
applied per-sample in the dataloader:

    wav  : torch.Tensor of shape (16000,) — 1 s @ 16 kHz
    mel  : torch.Tensor of shape (1, 40, 101) — log-mel spectrogram

Augmentations:
    - time_shift        : random circular shift of the waveform (word position
                          jitter — the word can start anywhere in the 1 s clip)
    - mix_background_noise : add a random 1 s crop of real background noise
                          at a random SNR in [5, 20] dB (robustness for free)
    - spec_augment      : SpecAugment-style masking of time bands and
                          frequency bands on the log-mel (forces the net to
                          use the whole utterance, not one lucky frame)
"""

import random

import torch


def time_shift(wav: torch.Tensor, max_shift: int = 1600) -> torch.Tensor:
    """Circularly shift the waveform by up to ±max_shift samples (±0.1 s)."""
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return wav
    return torch.roll(wav, shifts=shift, dims=0)


def _random_crop_1s(noise: torch.Tensor, n: int = 16000) -> torch.Tensor:
    if noise.numel() <= n:
        seg = noise[:n]
    else:
        start = random.randint(0, noise.numel() - n)
        seg = noise[start:start + n]
    return seg


def mix_background_noise(wav: torch.Tensor, noise: torch.Tensor,
                         snr_db: float | None = None) -> torch.Tensor:
    """Add background noise at a random SNR between 5 and 20 dB."""
    if snr_db is None:
        snr_db = random.uniform(5.0, 20.0)
    noise_seg = _random_crop_1s(noise).to(wav.dtype)
    wav_power = wav.pow(2).mean().clamp_min(1e-10)
    noise_power = noise_seg.pow(2).mean().clamp_min(1e-10)
    snr_linear = 10.0 ** (snr_db / 10.0)
    scale = torch.sqrt(wav_power / (snr_linear * noise_power))
    return wav + scale * noise_seg


def spec_augment(mel: torch.Tensor, time_mask: int = 25, freq_mask: int = 8,
                 n_time: int = 2, n_freq: int = 2) -> torch.Tensor:
    """Mask random time bands and frequency bands (SpecAugment, Park et al.)."""
    mel = mel.clone()
    _, n_mels, n_frames = mel.shape
    for _ in range(n_time):
        w = random.randint(0, time_mask)
        t0 = random.randint(0, max(0, n_frames - w))
        mel[:, :, t0:t0 + w] = mel.mean()
    for _ in range(n_freq):
        h = random.randint(0, freq_mask)
        f0 = random.randint(0, max(0, n_mels - h))
        mel[:, f0:f0 + h, :] = mel.mean()
    return mel


def augment_waveform(wav: torch.Tensor, noise_bank: list[torch.Tensor],
                     p_noise: float = 0.5) -> torch.Tensor:
    """Full waveform-stage augmentation: time shift + optional noise mix."""
    wav = time_shift(wav)
    if noise_bank and random.random() < p_noise:
        wav = mix_background_noise(wav, random.choice(noise_bank))
    return wav
