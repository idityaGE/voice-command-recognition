# Models

## `keyword_cnn.pt` — KeywordCNN (trained ✅)

**Architecture** (see `src/models.py`):

| Layer | Output shape |
|---|---|
| Input (log-mel spectrogram) | 1 × 40 × 101 |
| Conv2d(1→64, 3×3) + BN + ReLU + MaxPool(2×2) + Dropout(0.25) | 64 × 20 × 50 |
| Conv2d(64→128, 3×3) + BN + ReLU + MaxPool(2×2) + Dropout(0.25) | 128 × 10 × 25 |
| AdaptiveAvgPool(4×8) + Flatten | 4096 |
| Linear(4096→256) + ReLU + Dropout(0.5) | 256 |
| Linear(256→35) | 35 (logits) |

- Trainable parameters: **~1.13M**
- File size: ~0.9 MB
- Training: Adam (lr 1e-3), batch 256, cross-entropy, 15 epochs, CPU.
  Best checkpoint kept by validation accuracy.

**Measured metrics** (official test split, 11,005 clips):

| Metric | Value |
|---|---|
| Test accuracy | **63.33%** (11,005 clips) |
| Best validation accuracy | **65.86%** |

Per-class precision/recall: `results/metrics.json`.
Confusion matrix: `results/confusion_matrix.png`.

## `keyword_cnn_custom.pt` — fine-tuned on your voice (optional)

Created by `src/finetune_custom.py` after you record clips with
`src/record_custom.py`. Same architecture; the final layer is extended with
one neuron per new word. Not committed by default — it's personal to you.

## TinyTransformer (untrained — extension idea)

`src/models.py` also defines a 243,971-parameter transformer that treats
each spectrogram time-frame as a token. It is included for experimentation
but was not trained for this report; training it the same way as the CNN
(`train.py` accepts any model with minor edits) is a natural follow-up.
