# 🎙️ Voice Command Recognition System

A deep-learning **keyword spotting** system: it listens to 1-second voice
clips and recognizes which of 35 command words was spoken
("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", …).

Built as a deep-learning course project. The core idea: turn audio into
**log-mel spectrogram images** and train a small **convolutional neural
network** to read them — plus a classical baseline, a noise-robustness
study, and fine-tuning on your own voice.

> **Results (measured, not estimated):**
> - CNN test accuracy: **63.33%**
> - MFCC + SVM baseline test accuracy: **22.66%** (trained on a subset — see below)
> - Accuracy with background noise: clean **63.3%** → 0 dB SNR **18.2%**
>
> Full numbers: [`results/metrics.json`](results/metrics.json).
> New to the ideas? Start with [`FIRST_PRINCIPLES.md`](FIRST_PRINCIPLES.md).

---

## Repository tour

```
voice-command-recognition/
├── src/
│   ├── features.py          # waveform -> log-mel spectrogram
│   ├── dataset.py           # Speech Commands dataset (official splits)
│   ├── models.py            # KeywordCNN (+ optional TinyTransformer)
│   ├── train.py             # train the CNN
│   ├── evaluate.py          # test accuracy, confusion matrix, per-class scores
│   ├── eda.py               # class-balance + waveform/spectrogram plots
│   ├── baseline_svm.py      # classical baseline: MFCC + linear SVM
│   ├── noise_experiment.py  # accuracy vs background-noise SNR
│   ├── record_custom.py     # record your own 1-sec voice clips (needs mic)
│   └── finetune_custom.py   # fine-tune the CNN on your clips
├── app/
│   └── streamlit_app.py     # live demo: mic recording / wav upload -> prediction
├── models/
│   ├── keyword_cnn.pt       # trained CNN weights (committed, ~1 MB)
│   └── MODELS.md            # architecture + measured metrics
├── results/
│   ├── metrics.json         # test accuracy, per-class precision/recall
│   ├── svm_baseline.json    # baseline results
│   ├── noise_results.json   # clean vs noisy accuracy
│   ├── confusion_matrix.png
│   ├── top_confusions.txt
│   └── plots/               # EDA + noise-robustness charts
├── presentation/
│   └── voice_command_recognition.pptx   # classroom slides
├── data/                    # dataset lives here (git-ignored, see below)
├── FIRST_PRINCIPLES.md      # the whole project explained from physics up
├── requirements.txt
└── README.md
```

---

## Setup (Windows / macOS / Linux)

**1. Python 3.10+** — check with `python --version`.

**2. Create a virtual environment and install dependencies:**

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> CPU-only machine? Install the smaller CPU wheels explicitly:
> `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`
> then `pip install -r requirements.txt`.
> Have an NVIDIA GPU? Plain `pip install -r requirements.txt` fetches the CUDA
> build automatically and training will use it.

**3. Download the dataset** (Google Speech Commands v0.02, ~2.3 GB):

```bash
# from the repo root:
cd data
curl -L -o speech_commands_v0.02.tar.gz \
  http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar xzf speech_commands_v0.02.tar.gz
cd ..
```

You should now have `data/bed/`, `data/yes/`, … (35 word folders),
`data/_background_noise_/`, `data/validation_list.txt`,
`data/testing_list.txt`.

---

## Running the project

All commands run from the repo root with the venv activated.

```bash
# 1. Look at the data (saves plots to results/plots/)
python src/eda.py --data data

# 2. Train the CNN (~2-3 h on CPU, ~15 min on a GPU)
python src/train.py --data data --epochs 15 --out models

#    ...or skip training and use the committed weights:
#    models/keyword_cnn.pt is already trained.

# 3. Evaluate on the official test split
python src/evaluate.py --data data --ckpt models/keyword_cnn.pt

# 4. Classical baseline for comparison (subset documented in the script)
python src/baseline_svm.py --data data

# 5. Noise robustness experiment
python src/noise_experiment.py --data data --ckpt models/keyword_cnn.pt

# 6. Live demo (opens in your browser)
streamlit run app/streamlit_app.py
```

### Make it recognize YOUR voice

```bash
pip install sounddevice            # mic recording (bundles PortAudio)
python src/record_custom.py --word stop --n 20     # record 20 clips
python src/record_custom.py --word "ruk jao" --n 20
python src/finetune_custom.py --data data --custom-dir data/custom \
    --ckpt models/keyword_cnn.pt --epochs 10
# tick "Use my fine-tuned model" in the Streamlit sidebar
```

---

## How it works (30-second version)

1. Each 1-second clip (16,000 samples) is converted to a **log-mel
   spectrogram**: a 40×101 "image" of pitch-over-time, warped to match
   human hearing.
2. A small **CNN** (~1.1M params: 2 conv blocks + dense head) learns visual
   patterns in these images — the hiss of "six", the burst of "two".
3. **Softmax + cross-entropy** turn the 35 outputs into probabilities and a
   training signal; **Adam** optimizes for 15 epochs.
4. Evaluated on the held-out test split, compared against an MFCC+SVM
   baseline, and stress-tested with real background noise.

The long version: [`FIRST_PRINCIPLES.md`](FIRST_PRINCIPLES.md).

## Model

`KeywordCNN`: Conv(1→64) → BN → ReLU → MaxPool → Dropout →
Conv(64→128) → BN → ReLU → MaxPool → Dropout →
AdaptiveAvgPool(4×8) → Linear(4096→256) → ReLU → Dropout(0.5) →
Linear(256→35). Full metrics in [`models/MODELS.md`](models/MODELS.md).

A `TinyTransformer` alternative is defined in `src/models.py` for
experimentation (untrained — training it is left as an extension).

## Notes & limitations

- The dataset is mostly North-American accents; accuracy on other accents
  is lower — that's exactly what `finetune_custom.py` is for.
- Similar-sounding words ("three"/"tree", "four"/"forward") are the main
  confusion pairs — see `results/top_confusions.txt`.
- The SVM baseline trains on a stratified 8,000-clip subset (documented in
  `results/svm_baseline.json`) because a full SVM fit is slow; the CNN uses
  the full training split.
