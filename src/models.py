"""Model definitions.

KeywordCNN   -- the main model: a small 2D CNN over log-mel spectrograms.
TinyTransformer -- an optional alternative (sequence model over time frames).
                   Included for comparison; training it is optional.
"""

import math

import torch
import torch.nn as nn


class KeywordCNN(nn.Module):
    """Small CNN for keyword spotting (~1.2M parameters).

    Input : (B, 1, 40, 101) log-mel spectrogram
    Output: (B, 35) unnormalized class scores (logits)

    Two conv blocks learn local time-frequency patterns (e.g. the "sss"
    hiss of "six" vs the "tuh" burst of "two"); a light adaptive pool keeps
    enough time resolution to tell similar words apart ("four" vs "forward");
    a small dense head classifies.
    """

    def __init__(self, num_classes: int = 35, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (1, 40, 101) -> (64, 20, 50)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            # Block 2: (64, 20, 50) -> (128, 10, 25)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            # Keep time resolution: (128, 10, 25) -> (128, 4, 8).
            # Collapsing time too aggressively (e.g. to 2x2) throws away the
            # temporal detail that distinguishes confusable words.
            nn.AdaptiveAvgPool2d((4, 8)),   # -> 128*4*8 = 4096 features
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class TinyTransformer(nn.Module):
    """Tiny transformer baseline (~240k parameters, OPTIONAL - untrained by default).

    Treats each time frame of the spectrogram (a 40-dim mel vector) as one
    token in a sequence of ~101 tokens, adds positional information, runs a
    small transformer encoder, then averages over time and classifies.
    """

    def __init__(self, num_classes: int = 35, d_model: int = 96,
                 nhead: int = 4, num_layers: int = 3, dropout: float = 0.1,
                 max_frames: int = 128):
        super().__init__()
        self.proj = nn.Linear(40, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_frames, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 40, T) -> (B, T, 40)
        x = x.squeeze(1).transpose(1, 2)
        t = x.shape[1]
        x = self.proj(x) + self.pos[:, :t, :]
        x = self.dropout(x)
        x = self.encoder(x)          # (B, T, d_model)
        x = x.mean(dim=1)            # average over time
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name, m in [("KeywordCNN", KeywordCNN()), ("TinyTransformer", TinyTransformer())]:
        print(f"{name}: {count_parameters(m):,} trainable parameters")
        out = m(torch.randn(2, 1, 40, 101))
        print(f"  output shape: {tuple(out.shape)}")
