"""Exploratory data analysis: class balance + waveform/spectrogram examples.

Usage:
    python eda.py --data ../data --out ../results/plots
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import SpeechCommandsDataset
from features import load_waveform, waveform_to_logmel

EXAMPLE_WORDS = ["yes", "no", "stop", "go"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../results/plots")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = SpeechCommandsDataset(args.data, "train")
    print(f"train clips: {len(ds):,}")

    # --- class distribution ---
    counts = np.zeros(len(ds.classes), dtype=int)
    for _, label in ds.items:
        counts[label] += 1
    fig, ax = plt.subplots(figsize=(10, 8))
    order = np.argsort(counts)
    ax.barh([ds.classes[i] for i in order], counts[order])
    ax.set_xlabel("number of 1-sec clips (train split)")
    ax.set_title(f"Class balance: {len(ds.classes)} words "
                 f"(min {counts.min()}, max {counts.max()})")
    fig.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=120)
    plt.close(fig)
    print(f"class counts: min={counts.min()} max={counts.max()} "
          f"(imbalanced ~{counts.max()/counts.min():.1f}x)")

    # --- waveform + spectrogram examples ---
    fig, axes = plt.subplots(len(EXAMPLE_WORDS), 2,
                             figsize=(12, 3 * len(EXAMPLE_WORDS)))
    for row, word in enumerate(EXAMPLE_WORDS):
        path = next(p for p, l in ds.items
                    if ds.classes[l] == word)
        wav = load_waveform(path).numpy()
        spec = waveform_to_logmel(torch.from_numpy(wav)).squeeze(0).numpy()
        axes[row, 0].plot(wav, linewidth=0.5)
        axes[row, 0].set_title(f'waveform: "{word}"')
        axes[row, 0].set_xlim(0, 16000)
        im = axes[row, 1].imshow(spec, origin="lower", aspect="auto",
                                 cmap="magma")
        axes[row, 1].set_title(f'log-mel spectrogram: "{word}" (40 mels x time)')
        fig.colorbar(im, ax=axes[row, 1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "waveform_spectrogram_examples.png", dpi=120)
    plt.close(fig)
    print(f"saved plots to {out}")


if __name__ == "__main__":
    main()
