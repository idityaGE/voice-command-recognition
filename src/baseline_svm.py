"""Classical baseline: MFCC features + linear SVM (scikit-learn).

No deep learning here: each clip is reduced to 20 MFCC coefficients averaged
over time (a 20-dim vector), then a linear SVM classifies. Training an SVM on
all 105k clips is slow, so we use a stratified subset (default 8k train clips)
-- the subset size is reported honestly alongside the accuracy.

Usage:
    python baseline_svm.py --data ../data --n-train 8000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from dataset import SpeechCommandsDataset
from features import load_waveform, SAMPLE_RATE

_mfcc = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE, n_mfcc=20,
    melkwargs={"n_fft": 512, "win_length": 400, "hop_length": 160,
               "n_mels": 40})


def mfcc_mean(path: str) -> np.ndarray:
    wav = load_waveform(path).unsqueeze(0)          # (1, 16000)
    m = _mfcc(wav).squeeze(0)                        # (20, ~101)
    return m.mean(dim=1).numpy()                     # (20,) average over time


def featurize(items):
    X = np.stack([mfcc_mean(p) for p, _ in items])
    y = np.array([l for _, l in items])
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()

    train_ds = SpeechCommandsDataset(args.data, "train")
    test_ds = SpeechCommandsDataset(args.data, "test")

    # stratified subset of the training split (keeps class proportions)
    idx = np.arange(len(train_ds))
    labels = np.array([l for _, l in train_ds.items])
    sss = StratifiedShuffleSplit(n_splits=1, train_size=args.n_train,
                                 random_state=42)
    sub_idx, _ = next(sss.split(idx, labels))
    sub_items = [train_ds.items[i] for i in sorted(sub_idx)]
    print(f"SVM train subset: {len(sub_items):,} clips "
          f"(of {len(train_ds):,} total -- documented, not hidden)")

    print("extracting MFCC features (train subset)...")
    Xtr, ytr = featurize(sub_items)
    print("extracting MFCC features (full test split)...")
    Xte, yte = featurize(test_ds.items)

    scaler = StandardScaler().fit(Xtr)
    clf = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    clf.fit(scaler.transform(Xtr), ytr)
    acc = clf.score(scaler.transform(Xte), yte)
    print(f"SVM (MFCC) TEST ACCURACY on {len(yte):,} clips: {acc:.4f}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    with open(out / "svm_baseline.json", "w") as f:
        json.dump({"model": "LinearSVC on 20-dim mean-MFCC",
                   "n_train_subset": len(sub_items),
                   "n_train_total": len(train_ds),
                   "n_test": len(yte),
                   "test_accuracy": acc}, f, indent=2)


if __name__ == "__main__":
    main()
