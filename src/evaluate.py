"""Evaluate a trained checkpoint on the official test split.

Usage:
    python evaluate.py --data ../data --ckpt ../models/keyword_cnn.pt

Writes:
    results/metrics.json            (accuracy, per-class precision/recall)
    results/confusion_matrix.png    (35x35 heatmap)
    results/top_confusions.txt      (most confused word pairs)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from dataset import SpeechCommandsDataset
from models import KeywordCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--ckpt", default="../models/keyword_cnn.pt")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default="../results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    classes = ckpt["classes"]

    test_ds = SpeechCommandsDataset(args.data, "test")
    assert test_ds.classes == classes, "checkpoint classes != dataset classes"
    loader = DataLoader(test_ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)
    print(f"test clips: {len(test_ds):,}")

    model = KeywordCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            all_pred.extend(logits.argmax(1).cpu().tolist())
            all_true.extend(yb.tolist())

    acc = float(np.mean(np.array(all_pred) == np.array(all_true)))
    print(f"TEST ACCURACY: {acc:.4f}")

    report = classification_report(all_true, all_pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    per_class = {c: {"precision": report[c]["precision"],
                     "recall": report[c]["recall"],
                     "f1": report[c]["f1-score"],
                     "support": report[c]["support"]} for c in classes}

    cm = confusion_matrix(all_true, all_pred, labels=list(range(len(classes))))

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump({"test_accuracy": acc, "n_test": len(test_ds),
                   "checkpoint_val_acc": ckpt.get("val_acc"),
                   "params": ckpt.get("params"),
                   "per_class": per_class}, f, indent=2)

    # --- confusion matrix heatmap ---
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (test accuracy {acc:.2%})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    # --- most confused pairs (off-diagonal, symmetric) ---
    pairs = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            n = int(cm[i, j] + cm[j, i])
            if n:
                pairs.append((n, classes[i], classes[j]))
    pairs.sort(reverse=True)
    with open(out / "top_confusions.txt", "w") as f:
        f.write("most confused word pairs (true<->predicted), test set:\n")
        for n, a, b in pairs[:15]:
            f.write(f"  {a:>10} <-> {b:<10} : {n} confusions\n")
    print("top confused pairs:")
    for n, a, b in pairs[:8]:
        print(f"  {a} <-> {b}: {n}")
    print(f"wrote {out/'metrics.json'}, confusion_matrix.png, top_confusions.txt")


if __name__ == "__main__":
    main()
