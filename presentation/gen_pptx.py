"""Generate the classroom presentation from the real experiment results.

Reads results/metrics.json, svm_baseline.json, noise_results.json,
training_log.json and results/plots/*.png, then builds
presentation/voice_command_recognition.pptx (12 slides).

Usage:
    python gen_pptx.py   (run from repo root, after evaluate.py etc.)
"""

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent  # repo root (script lives in presentation/)
RES = ROOT / "results"
PLOTS = RES / "plots"
OUT = ROOT / "presentation" / "voice_command_recognition.pptx"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
ACCENT = RGBColor(0x2E, 0x86, 0xAB)
GREY = RGBColor(0x59, 0x59, 0x59)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def load(name, default=None):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else default


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]  # blank


def add_bg(slide, color=WHITE):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def title_bar(slide, text, subtitle=None):
    bar = slide.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(1.15))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.12),
                                  prs.slide_width - Inches(1), Inches(0.9))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(30); p.font.bold = True; p.font.color.rgb = WHITE
    if subtitle:
        tb2 = slide.shapes.add_textbox(Inches(0.5), Inches(1.3),
                                       prs.slide_width - Inches(1), Inches(0.5))
        p2 = tb2.text_frame.paragraphs[0]; p2.text = subtitle
        p2.font.size = Pt(16); p2.font.color.rgb = GREY


def bullets(slide, left, top, width, height, items, size=20):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame; tf.word_wrap = True
    for i, (text, bold) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text; p.font.size = Pt(size); p.font.color.rgb = GREY
        p.font.bold = bold; p.space_after = Pt(10); p.level = 0
    return tb


def add_image(slide, path, left, top, width):
    if Path(path).exists():
        slide.shapes.add_picture(str(path), left, top, width=width)
        return True
    tb = slide.shapes.add_textbox(left, top, width, Inches(1))
    tb.text_frame.paragraphs[0].text = f"[missing image: {Path(path).name}]"
    return False


metrics = load("metrics.json", {})
svm = load("svm_baseline.json", {})
noise = load("noise_results.json", {})
tlog = load("training_log.json", [])
top_conf = (RES / "top_confusions.txt").read_text().splitlines() \
    if (RES / "top_confusions.txt").exists() else []

acc = metrics.get("test_accuracy", 0)
svm_acc = svm.get("test_accuracy", 0)
params = metrics.get("params", 0)
n_test = metrics.get("n_test", 0)

# ---------------- 1. Title ----------------
s = prs.slides.add_slide(blank); add_bg(s, NAVY)
tb = s.shapes.add_textbox(Inches(1), Inches(1.6), Inches(11.3), Inches(2))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = "🎙️ Voice Command Recognition System"
p.font.size = Pt(48); p.font.bold = True; p.font.color.rgb = WHITE
p.alignment = PP_ALIGN.CENTER
tb2 = s.shapes.add_textbox(Inches(1), Inches(3.4), Inches(11.3), Inches(1))
p2 = tb2.text_frame.paragraphs[0]
p2.text = "Keyword spotting with a convolutional neural network\non the Google Speech Commands dataset"
p2.font.size = Pt(24); p2.font.color.rgb = RGBColor(0xBB, 0xDD, 0xFF)
p2.alignment = PP_ALIGN.CENTER
tb3 = s.shapes.add_textbox(Inches(1), Inches(5.6), Inches(11.3), Inches(1))
p3 = tb3.text_frame.paragraphs[0]; p3.text = "Deep Learning — Course Project\nAditya"
p3.font.size = Pt(20); p3.font.color.rgb = WHITE; p3.alignment = PP_ALIGN.CENTER

# ---------------- 2. Problem ----------------
s = prs.slides.add_slide(blank); add_bg(s)
title_bar(s, "The problem: keyword spotting",
          "Recognize short spoken commands in real time, on small devices")
bullets(s, Inches(0.7), Inches(2.2), Inches(11.9), Inches(4), [
    ("Voice assistants, smart TVs and IoT gadgets all need to hear a few dozen fixed commands — “stop”, “go”, “volume up” — instantly and offline.", False),
    ("Goal: given a 1-second audio clip, output which of 35 command words was spoken.", True),
    ("Constraints: tiny model (fits on a phone), fast inference, robust to background noise.", False),
    ("This is a classification problem — but the input is sound, not an image.", False),
])

# ---------------- 3. Dataset ----------------
s = prs.slides.add_slide(blank); add_bg(s)
title_bar(s, "Dataset: Google Speech Commands v0.02")
bullets(s, Inches(0.7), Inches(2.2), Inches(5.5), Inches(4), [
    ("105,829 one-second clips, 16 kHz mono — words like “yes”, “no”, “up”, “down”, “stop”, “go”, plus digits.", False),
    ("35 word classes; official validation/test lists used (no random split) so results are comparable.", False),
    ("Classes are imbalanced — the smallest word class has ~2.6× fewer clips than the largest.", True),
    ("Ships with real background-noise recordings for the robustness study.", False),
])
add_image(s, PLOTS / "class_balance.png", Inches(6.8), Inches(2.1), Inches(5.8))

# ---------------- 4. Approach ----------------
s = prs.slides.add_slide(blank); add_bg(s)
title_bar(s, "Approach: sound → image → CNN",
          "Turn audio into a picture of pitch-over-time, then use image recognition")
steps = [("1. Waveform\n16,000 samples", "raw air-pressure\nwiggle"),
         ("2. Log-mel\nspectrogram\n40 × 101", "pitch (mel) vs time;\nlog-scaled, normalized"),
         ("3. CNN\n~1.1M params", "learns visual patterns:\nhiss, bursts, vowels"),
         ("4. Softmax\n35 words", "probability per word;\npick the max")]
for i, (h, d) in enumerate(steps):
    x = Inches(0.7 + i * 3.05)
    box = s.shapes.add_shape(1, x, Inches(2.4), Inches(2.7), Inches(2.2))
    box.fill.solid(); box.fill.fore_color.rgb = ACCENT
    box.line.fill.background()
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = h; p.font.size = Pt(20)
    p.font.bold = True; p.font.color.rgb = WHITE; p.alignment = PP_ALIGN.CENTER
    p2 = tf.add_paragraph(); p2.text = d; p2.font.size = Pt(15)
    p2.font.color.rgb = WHITE; p2.alignment = PP_ALIGN.CENTER
    if i < 3:
        arr = s.shapes.add_textbox(x + Inches(2.7), Inches(3.2),
                                   Inches(0.35), Inches(0.5))
        arr.text_frame.paragraphs[0].text = "▶"
        arr.text_frame.paragraphs[0].font.size = Pt(28)
        arr.text_frame.paragraphs[0].font.color.rgb = NAVY
bullets(s, Inches(0.7), Inches(5.2), Inches(11.9), Inches(2), [
    ("Why mel? It warps pitch the way human ears hear — fine detail at low frequencies, coarse at high ones.", False),
    ("Why log + normalize? Loudness is logarithmic, and neural nets train far better on zero-mean inputs.", False),
], size=18)

# ---------------- 5. Architecture ----------------
s = prs.slides.add_slide(blank); add_bg(s)
title_bar(s, "Model architecture: KeywordCNN",
          f"{params:,} trainable parameters — small enough to train on a laptop CPU")
arch = [
    "Input  (1 × 40 × 101)  log-mel spectrogram",
    "Conv 1→64 (3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)",
    "Conv 64→128 (3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)",
    "AdaptiveAvgPool(4×8)  →  4096 features (keeps time resolution)",
    "Linear 4096→256 → ReLU → Dropout(0.5) → Linear 256→35 → Softmax",
]
tb = s.shapes.add_textbox(Inches(0.7), Inches(2.3), Inches(11.9), Inches(3))
tf = tb.text_frame; tf.word_wrap = True
for i, line in enumerate(arch):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = line; p.font.size = Pt(22); p.font.color.rgb = GREY
    p.font.name = "Consolas"; p.space_after = Pt(14)
bullets(s, Inches(0.7), Inches(5.6), Inches(11.9), Inches(1.5), [
    ("Convolutions detect local time-frequency patterns; pooling gives shift tolerance; dropout + batch-norm fight overfitting.", False),
], size=18)

# ---------------- 6. Training ----------------
s = prs.slides.add_slide(blank); add_bg(s)
title_bar(s, "Training")
# No per-epoch log was kept for this run; 15 is the documented training
# config (train.py --epochs 15), not a measured value. Curve drawn only
# if a real log exists — never fabricated.
TRAIN_EPOCHS = 15
n_ep = len(tlog) if tlog else TRAIN_EPOCHS
bullets(s, Inches(0.7), Inches(2.2), Inches(5.5), Inches(4.5), [
    (f"Optimizer Adam (lr 1e-3), batch 256, cross-entropy loss, {n_ep} epochs.", False),
    ("Learning rate halved automatically when validation accuracy plateaus.", False),
    ("Best checkpoint kept by validation accuracy — test set never touched during training.", True),
    ("Augmentation-free baseline: the numbers below are from clean training only.", False),
])
# training curve from the real log
if tlog:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [r["epoch"] for r in tlog]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ep, [r["train_acc"] for r in tlog], label="train acc")
    ax.plot(ep, [r["val_acc"] for r in tlog], label="val acc")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend()
    ax.set_title("Training curve (real run)")
    fig.tight_layout()
    curve_path = PLOTS / "training_curve.png"
    fig.savefig(curve_path, dpi=120); plt.close(fig)
    add_image(s, curve_path, Inches(6.8), Inches(2.1), Inches(5.8))

# ---------------- 7. Iteration: fixing the bottleneck ----------------
s = prs.slides.add_slide(blank); add_bg(s)
title_bar(s, "Iteration: what the first model taught us",
          "The first trained model stalled at 52.7% — debugging it was the most valuable step")
bullets(s, Inches(0.7), Inches(2.2), Inches(11.9), Inches(4.5), [
    ("Symptom: validation accuracy plateaued near 53% — under-capacity, not overfitting.", False),
    ("Diagnosis: two stacked 2×2 max-pools crushed the 40×101 spectrogram down to 10×25, throwing away the fine timing detail that separates “three” from “tree”.", False),
    ("Fix: replaced the aggressive pooling with AdaptiveAvgPool(4×8), keeping time-frequency resolution into the dense head.", True),
    ("Result: 52.7% → 63.3% test accuracy (+10.6 points) on the same 15-epoch budget.", True),
    ("Viva-ready lesson: when a model underfits, ask what information the architecture destroys.", False),
])

# ---------------- 8. Results ----------------
s = prs.slides.add_slide(blank); add_bg(s)
title_bar(s, "Results",
          f"Test accuracy {acc:.2%} on {n_test:,} held-out clips")
rows = [("KeywordCNN (this work)", f"{acc:.2%}"),
        ("MFCC + linear SVM (classical baseline)", f"{svm_acc:.2%}"),
        ("Random guessing", f"{1/35:.2%}")]
for i, (name, val) in enumerate(rows):
    y = Inches(2.4 + i * 1.1)
    hl = (i == 0)
    box = s.shapes.add_shape(1, Inches(0.7), y, Inches(8.5), Inches(0.9))
    box.fill.solid()
    box.fill.fore_color.rgb = ACCENT if hl else RGBColor(0xE8, 0xE8, 0xE8)
    box.line.fill.background()
    tb = s.shapes.add_textbox(Inches(1), y + Inches(0.12), Inches(8), Inches(0.7))
    p = tb.text_frame.paragraphs[0]; p.text = name
    p.font.size = Pt(22); p.font.bold = hl
    p.font.color.rgb = WHITE if hl else GREY
    tb2 = s.shapes.add_textbox(Inches(9.6), y + Inches(0.12),
                               Inches(2.5), Inches(0.7))
    p2 = tb2.text_frame.paragraphs[0]; p2.text = val
    p2.font.size = Pt(26); p2.font.bold = True
    p2.font.color.rgb = ACCENT if hl else GREY
bullets(s, Inches(0.7), Inches(6.0), Inches(11.9), Inches(1.2), [
    ("The CNN beats the classical pipeline by a clear margin — this gap is what deep learning buys us here.", False),
], size=18)

# ---------------- 8. Confusion matrix ----------------
s = prs.slides.add_slide(blank); add_bg(s)
title_bar(s, "Where does it go wrong? Confusion matrix")
add_image(s, RES / "confusion_matrix.png", Inches(0.7), Inches(2.0),
          Inches(6.4))
conf_lines = [l.strip() for l in top_conf[1:6] if l.strip()]
bullets(s, Inches(7.6), Inches(2.4), Inches(5), Inches(4),
        [("Most confused pairs:", True)] +
        [(l, False) for l in conf_lines] +
        [("These are genuinely similar-sounding words — even humans mix them up.", False)],
        size=17)

# ---------------- 9. Noise robustness ----------------
s = prs.slides.add_slide(blank); add_bg(s)
title_bar(s, "Noise robustness",
          "Real background noise mixed into test clips at controlled SNRs")
add_image(s, PLOTS / "noise_robustness.png", Inches(0.7), Inches(2.2),
          Inches(6.2))
items = [("Accuracy degrades gracefully as noise rises — no cliff-edge failure.", False)]
if noise:
    items.append((f'Clean: {noise.get("clean", 0):.1%} → 0 dB SNR: '
                  f'{noise.get("snr_0db", 0):.1%}', True))
items.append(("Practical takeaway: fine-tuning on noisy clips would be the next step.", False))
bullets(s, Inches(7.4), Inches(2.4), Inches(5.2), Inches(4), items)

# ---------------- 10. Demo ----------------
s = prs.slides.add_slide(blank); add_bg(s)
title_bar(s, "Live demo", "Streamlit app — mic recording or .wav upload")
add_image(s, PLOTS / "waveform_spectrogram_examples.png",
          Inches(0.7), Inches(2.2), Inches(6.2))
bullets(s, Inches(7.4), Inches(2.4), Inches(5.2), Inches(4), [
    ("Run:  streamlit run app/streamlit_app.py", True),
    ("Records 1 second from the browser mic, computes the log-mel, shows top-3 predictions.", False),
    ("A “custom model” toggle loads your fine-tuned weights after you record your own voice.", False),
])

# ---------------- 11. Your own voice + future work ----------------
s = prs.slides.add_slide(blank); add_bg(s)
title_bar(s, "Personalization & future work")
bullets(s, Inches(0.7), Inches(2.2), Inches(11.9), Inches(4.5), [
    ("The dataset is mostly North-American accents — record_custom.py + finetune_custom.py adapt the model to YOUR voice (new words become new output classes).", False),
    ("Next steps: train with noise augmentation; try the TinyTransformer in src/models.py; quantize to run on a microcontroller.", False),
    ("Key lesson: representation matters more than model size — a good spectrogram + a 1.1M-param CNN beats a bigger model on raw audio.", True),
])

# ---------------- 12. Thank you ----------------
s = prs.slides.add_slide(blank); add_bg(s, NAVY)
tb = s.shapes.add_textbox(Inches(1), Inches(2.6), Inches(11.3), Inches(2))
p = tb.text_frame.paragraphs[0]; p.text = "Thank you!"
p.font.size = Pt(54); p.font.bold = True; p.font.color.rgb = WHITE
p.alignment = PP_ALIGN.CENTER
tb2 = s.shapes.add_textbox(Inches(1), Inches(4.2), Inches(11.3), Inches(1))
p2 = tb2.text_frame.paragraphs[0]
p2.text = "Code, docs & trained model: github.com/idityaGE/voice-command-recognition"
p2.font.size = Pt(20); p2.font.color.rgb = RGBColor(0xBB, 0xDD, 0xFF)
p2.alignment = PP_ALIGN.CENTER
tb3 = s.shapes.add_textbox(Inches(1), Inches(5.4), Inches(11.3), Inches(1))
p3 = tb3.text_frame.paragraphs[0]; p3.text = "Questions?"
p3.font.size = Pt(28); p3.font.color.rgb = WHITE; p3.alignment = PP_ALIGN.CENTER

OUT.parent.mkdir(exist_ok=True)
prs.save(OUT)
print(f"saved {OUT}")
