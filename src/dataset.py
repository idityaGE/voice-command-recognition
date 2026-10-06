"""PyTorch dataset for Google Speech Commands v0.02.

Layout expected under <data_root>/ :
    bed/0a7c2a8d_nohash_0.wav, ...      (one folder per spoken word)
    _background_noise_/...              (excluded from classification)
    validation_list.txt                 (e.g. "bed/0a7c2a8d_nohash_0.wav")
    testing_list.txt

We use the *official* validation/test lists instead of a random split, so our
numbers are comparable with published results on this dataset.
"""

import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from features import load_waveform, waveform_to_logmel
from augment import augment_waveform, spec_augment

IGNORE_DIRS = {"_background_noise_", "custom"}


def discover_classes(data_root: str) -> list[str]:
    """Class names = subfolder names, sorted, minus the noise folder."""
    names = sorted(
        d.name for d in Path(data_root).iterdir()
        if d.is_dir() and d.name not in IGNORE_DIRS
    )
    assert len(names) == 35, f"expected 35 word folders, found {len(names)}"
    return names


def _read_list(data_root: str, filename: str) -> set[str]:
    with open(os.path.join(data_root, filename)) as f:
        return {line.strip() for line in f if line.strip()}


class SpeechCommandsDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train",
                 augment: bool = False):
        assert split in ("train", "val", "test")
        # Augmentation only ever applies to the training split.
        self.augment = augment and split == "train"
        self.classes = discover_classes(data_root)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        val = _read_list(data_root, "validation_list.txt")
        test = _read_list(data_root, "testing_list.txt")

        self.items: list[tuple[str, int]] = []
        for cls in self.classes:
            folder = Path(data_root) / cls
            for wav_path in sorted(folder.glob("*.wav")):
                rel = f"{cls}/{wav_path.name}"
                in_val, in_test = rel in val, rel in test
                if split == "train" and not (in_val or in_test):
                    self.items.append((str(wav_path), self.class_to_idx[cls]))
                elif split == "val" and in_val:
                    self.items.append((str(wav_path), self.class_to_idx[cls]))
                elif split == "test" and in_test:
                    self.items.append((str(wav_path), self.class_to_idx[cls]))

        # Preload background-noise recordings once for noise augmentation.
        self.noise_bank: list[torch.Tensor] = []
        if self.augment:
            noise_dir = Path(data_root) / "_background_noise_"
            for wav_path in sorted(noise_dir.glob("*.wav")):
                try:
                    self.noise_bank.append(load_waveform(str(wav_path)))
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        wav = load_waveform(path)
        if self.augment:
            wav = augment_waveform(wav, self.noise_bank)  # shift + noise mix
        logmel = waveform_to_logmel(wav)  # (1, 40, 101)
        if self.augment and random.random() < 0.5:
            logmel = spec_augment(logmel)  # mask time/freq bands
        return logmel, label
