"""Noise robustness experiment: how does background noise hurt accuracy?

For each test clip we mix in a random slice of real background noise from the
dataset's `_background_noise_` folder at a target SNR (signal-to-noise ratio),
then measure the trained CNN's accuracy. SNR = inf means the clean clips.

Usage:
    python noise_experiment.py --data ../data --ckpt ../models/keyword_cnn.pt

Writes results/noise_results.json + results/plots/noise_robustness.png
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

from dataset import SpeechCommandsDataset
from features import load_waveform, waveform_to_logmel, CLIP_SAMPLES, SAMPLE_RATE
from models import KeywordCNN

SNRS_DB = [None, 10, 5, 0]   # None = clean


def load_noise_bank(data_root: str) -> list[torch.Tensor]:
    """Long noise recordings concatenated into one big waveform."""
    import soundfile as sf
    wavs = []
    for p in sorted(Path(data_root, "_background_noise_").glob("*.wav")):
        wav_np, sr = sf.read(str(p), dtype="float32")
        wav = torch.from_numpy(wav_np)
        if wav.dim() > 1:
            wav = wav.mean(dim=1)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wavs.append(wav)
    return wavs


def mix_at_snr(clean: torch.Tensor, noise_bank: list[torch.Tensor],
               snr_db: float | None, rng: random.Random) -> torch.Tensor:
    if snr_db is None:
        return clean
    noise_src = rng.choice(noise_bank)
    start = rng.randrange(0, max(1, noise_src.numel() - CLIP_SAMPLES))
    noise = noise_src[start:start + CLIP_SAMPLES]
    if noise.numel() < CLIP_SAMPLES:                       # pad short noise
        noise = torch.cat([noise, torch.zeros(CLIP_SAMPLES - noise.numel())])
    # scale noise so that 10*log10(P_signal / P_noise) == snr_db
    p_sig = clean.pow(2).mean().clamp(min=1e-10)
    p_noise = noise.pow(2).mean().clamp(min=1e-10)
    scale = torch.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
    return clean + scale * noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--ckpt", default="../models/keyword_cnn.pt")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()
    rng = random.Random(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = KeywordCNN(num_classes=len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = SpeechCommandsDataset(args.data, "test")
    noise_bank = load_noise_bank(args.data)
    print(f"test clips: {len(test_ds):,}, noise files: {len(noise_bank)}")

    results = {}
    with torch.no_grad():
        for snr in SNRS_DB:
            correct, total = 0, 0
            for path, label in test_ds.items:
                clean = load_waveform(path)
                mixed = mix_at_snr(clean, noise_bank, snr, rng)
                logmel = waveform_to_logmel(mixed).unsqueeze(0).to(device)
                pred = model(logmel).argmax(1).item()
                correct += (pred == label)
                total += 1
            acc = correct / total
            key = "clean" if snr is None else f"snr_{snr}db"
            results[key] = acc
            print(f"  {key:>10}: accuracy {acc:.4f}")

    out = Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    with open(out / "noise_results.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["clean", "10 dB", "5 dB", "0 dB"]
    keys = ["clean", "snr_10db", "snr_5db", "snr_0db"]
    ax.bar(labels, [results[k] for k in keys])
    ax.set_ylim(0, 1)
    ax.set_ylabel("test accuracy")
    ax.set_title("Noise robustness: CNN accuracy vs background-noise SNR")
    for i, k in enumerate(keys):
        ax.text(i, results[k] + 0.02, f"{results[k]:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(out / "plots" / "noise_robustness.png", dpi=120)
    print(f"wrote noise_results.json + plots/noise_robustness.png")


if __name__ == "__main__":
    main()
