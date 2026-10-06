"""Fine-tune the trained CNN on your own recorded voice clips.

New words (not in the original 35) are APPENDED as new output classes; words
that match an existing class just get extra training data. To avoid forgetting
the original words, each epoch also replays a few original clips per class.

Usage:
    python finetune_custom.py --data ../data --custom-dir ../data/custom \\
        --ckpt ../models/keyword_cnn.pt --epochs 10

Writes: models/keyword_cnn_custom.pt  (+ prints per-word accuracy on your clips)
"""

import argparse
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dataset import SpeechCommandsDataset
from features import load_waveform, waveform_to_logmel
from models import KeywordCNN


class CustomDataset(Dataset):
    def __init__(self, custom_dir: str, class_to_idx: dict):
        self.items = []
        for word_dir in sorted(Path(custom_dir).iterdir()):
            if not word_dir.is_dir():
                continue
            for wav in sorted(word_dir.glob("*.wav")):
                self.items.append((str(wav), word_dir.name))
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, word = self.items[idx]
        return waveform_to_logmel(load_waveform(path)), self.class_to_idx[word]


def extend_head(model: KeywordCNN, old_classes: list[str],
                new_classes: list[str]) -> tuple[KeywordCNN, list[str]]:
    """Append new output neurons (randomly initialized) for unseen words."""
    added = [c for c in new_classes if c not in old_classes]
    if not added:
        return model, old_classes
    all_classes = old_classes + added
    new_model = KeywordCNN(num_classes=len(all_classes))
    new_model.features.load_state_dict(model.features.state_dict())
    # copy the hidden Linear layer as-is; grow the output layer, keeping old rows
    new_model.head[1].weight.data.copy_(model.head[1].weight.data)
    new_model.head[1].bias.data.copy_(model.head[1].bias.data)
    n_old = model.head[4].out_features
    with torch.no_grad():
        new_model.head[4].weight.data[:n_old].copy_(model.head[4].weight.data)
        new_model.head[4].bias.data[:n_old].copy_(model.head[4].bias.data)
    print(f"extended classifier: {len(old_classes)} -> {len(all_classes)} "
          f"classes (new: {added})")
    return new_model, all_classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--custom-dir", default="../data/custom")
    ap.add_argument("--ckpt", default="../models/keyword_cnn.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--replay-per-class", type=int, default=15,
                    help="original clips per class replayed each epoch")
    ap.add_argument("--out", default="../models/keyword_cnn_custom.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    custom_words = sorted(d.name for d in Path(args.custom_dir).iterdir()
                          if d.is_dir())
    assert custom_words, f"no word folders found in {args.custom_dir} -- run record_custom.py first"
    print(f"custom words: {custom_words}")

    model = KeywordCNN(num_classes=len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model, all_classes = extend_head(model, ckpt["classes"], custom_words)
    model.to(device)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    custom_ds = CustomDataset(args.custom_dir, class_to_idx)
    print(f"custom clips: {len(custom_ds)}")

    # replay buffer: a few original clips per old class (prevents forgetting)
    orig_ds = SpeechCommandsDataset(args.data, "train")
    by_class: dict[int, list] = {}
    for p, l in orig_ds.items:
        by_class.setdefault(l, []).append((p, l))
    rng = random.Random(42)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        replay = []
        for l, items in by_class.items():
            replay += rng.sample(items, min(args.replay_per_class, len(items)))
        # wrap raw (path, label) pairs into tensors on the fly
        model.train()
        tot_loss, correct, total = 0.0, 0, 0
        loader = DataLoader(custom_ds, batch_size=32, shuffle=True)
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            # mix in a few replay clips
            rb = rng.sample(replay, min(32, len(replay)))
            rx = torch.stack([waveform_to_logmel(load_waveform(p))
                              for p, _ in rb]).to(device)
            ry = torch.tensor([l for _, l in rb], device=device)
            xb, yb = torch.cat([xb, rx]), torch.cat([yb, ry])
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(xb)
        print(f"epoch {epoch}/{args.epochs}  loss {tot_loss/total:.4f} "
              f"acc {correct/total:.4f}")

    # quick check: accuracy on the custom clips themselves
    model.eval()
    correct = 0
    with torch.no_grad():
        for p, w in custom_ds.items:
            pred = model(waveform_to_logmel(load_waveform(p))
                         .unsqueeze(0).to(device)).argmax(1).item()
            correct += (pred == class_to_idx[w])
    print(f"accuracy on your {len(custom_ds)} custom clips: {correct/len(custom_ds):.2%}")

    os.makedirs(Path(args.out).parent, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "classes": all_classes},
               args.out)
    print(f"saved {args.out}  (classes: {all_classes})")


if __name__ == "__main__":
    main()
