"""Train the KeywordCNN on Google Speech Commands.

Usage:
    python train.py --data ../data --epochs 15 --batch-size 256 --out ../models
    python train.py --data ../data --epochs 40 --augment --weighted --out ../models

Flags --augment (time shift + background-noise mix + SpecAugment) and
--weighted (class-weighted loss) are the accuracy push: they make training
see a harder, more balanced version of the data.

Saves:
    models/keyword_cnn.pt      (best checkpoint by validation accuracy)
    results/training_log.json  (loss/accuracy per epoch)
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SpeechCommandsDataset
from models import KeywordCNN, count_parameters


def run_epoch(model, loader, criterion, optimizer, device, train: bool,
              amp: bool = False):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(train):
            if amp and train:
                with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                    logits = model(xb)
                    loss = criterion(logits, yb)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(xb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(xb)
    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", default="../models")
    ap.add_argument("--augment", action="store_true",
                    help="on-the-fly augmentation: time shift, background-noise "
                         "mixing, SpecAugment (train split only)")
    ap.add_argument("--weighted", action="store_true",
                    help="class-weighted cross-entropy to counter class imbalance")
    ap.add_argument("--amp", action="store_true",
                    help="bfloat16 mixed precision on CPU (~1.5x faster training)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | augment: {args.augment} | weighted: {args.weighted}")

    train_ds = SpeechCommandsDataset(args.data, "train", augment=args.augment)
    val_ds = SpeechCommandsDataset(args.data, "val")
    print(f"train: {len(train_ds):,} clips | val: {len(val_ds):,} clips | "
          f"classes: {len(train_ds.classes)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    model = KeywordCNN(num_classes=len(train_ds.classes)).to(device)
    print(f"parameters: {count_parameters(model):,}")

    weight = None
    if args.weighted:
        counts = torch.zeros(len(train_ds.classes))
        for _, y in train_ds.items:
            counts[y] += 1
        weight = (counts.sum() / counts / len(counts)).to(device)  # rarer -> larger
        print(f"class weights: min {weight.min():.2f} / max {weight.max():.2f}")
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    os.makedirs(args.out, exist_ok=True)
    results_dir = Path(args.out).parent / "results"
    results_dir.mkdir(exist_ok=True)

    best_val, log = 0.0, []
    ckpt_path = os.path.join(args.out, "keyword_cnn.pt")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                    device, True, amp=args.amp)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer,
                                    device, False)
        scheduler.step(va_acc)
        log.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                    "val_loss": va_loss, "val_acc": va_acc,
                    "lr": optimizer.param_groups[0]["lr"]})
        print(f"epoch {epoch:2d}/{args.epochs}  "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}  "
              f"({time.time()-t0:.0f}s elapsed)")
        if va_acc > best_val:
            best_val = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": train_ds.classes,
                "val_acc": va_acc,
                "params": count_parameters(model),
            }, ckpt_path)
            print(f"  -> saved new best ({va_acc:.4f}) to {ckpt_path}")

    with open(results_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"done. best val accuracy: {best_val:.4f} | total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
