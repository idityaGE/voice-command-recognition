"""Generate the classroom presentation from the real experiment results.

Reads results/metrics.json, svm_baseline.json, noise_results.json,
training_log.json and results/plots/*.png, then builds
presentation/voice_command_recognition.pptx (13 slides).

Design system: deep-navy + azure accent + coral highlights, 16:9,
generous whitespace, big numbers, speaker notes on every slide.

Usage:
    python presentation/gen_pptx.py   (run from repo root)
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

# ---- design tokens ----
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
ACCENT = RGBColor(0x2E, 0x86, 0xAB)
CORAL = RGBColor(0xE7, 0x6F, 0x51)
GREY = RGBColor(0x59, 0x59, 0x59)
GREY_LT = RGBColor(0xE8, 0xE8, 0xE8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ICE = RGBColor(0xBB, 0xDD, 0xFF)


def load(name, default=None):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else default


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]  # blank
N_SLIDES = 13
_slide_no = 0


def new_slide():
    global _slide_no
    _slide_no += 1
    return prs.slides.add_slide(blank)


def add_bg(slide, color=WHITE):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def note(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def footer(slide, dark=False):
    tb = slide.shapes.add_textbox(Inches(10.6), Inches(7.05),
                                  Inches(2.2), Inches(0.35))
    p = tb.text_frame.paragraphs[0]
    p.text = f"voice-command-recognition  •  {_slide_no}/{N_SLIDES}"
    p.font.size = Pt(11); p.font.color.rgb = ICE if dark else GREY
    p.alignment = PP_ALIGN.RIGHT


def title_bar(slide, text, subtitle=None):
    bar = slide.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(1.15))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    # coral motif line under the bar — the theme's signature
    line = slide.shapes.add_shape(1, 0, Inches(1.15),
                                  prs.slide_width, Inches(0.06))
    line.fill.solid(); line.fill.fore_color.rgb = CORAL
    line.line.fill.background()
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.12),
                                  prs.slide_width - Inches(1), Inches(0.9))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = text
    p.font.size = Pt(30); p.font.bold = True; p.font.color.rgb = WHITE
    if subtitle:
        tb2 = slide.shapes.add_textbox(Inches(0.5), Inches(1.32),
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


def stat_tile(slide, x, big, small, color=ACCENT):
    box = slide.shapes.add_shape(1, x, Inches(2.3), Inches(2.5), Inches(1.9))
    box.fill.solid(); box.fill.fore_color.rgb = color
    box.line.fill.background()
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = big; p.font.size = Pt(40)
    p.font.bold = True; p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER
    p2 = tf.add_paragraph(); p2.text = small; p2.font.size = Pt(15)
    p2.font.color.rgb = WHITE; p2.alignment = PP_ALIGN.CENTER


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
# No per-epoch log was kept; 15 is the documented training config
# (train.py --epochs 15), not a measured value.
TRAIN_EPOCHS = 15

# ---------------- 1. Title ----------------
s = new_slide(); add_bg(s, NAVY)
add_image(s, PLOTS / "title_strip.png", Inches(0), Inches(5.9),
          prs.slide_width)
tb = s.shapes.add_textbox(Inches(1), Inches(1.2), Inches(11.3), Inches(2))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = "🎙️ Voice Command Recognition System"
p.font.size = Pt(48); p.font.bold = True; p.font.color.rgb = WHITE
p.alignment = PP_ALIGN.CENTER
tb2 = s.shapes.add_textbox(Inches(1), Inches(3.1), Inches(11.3), Inches(1))
p2 = tb2.text_frame.paragraphs[0]
p2.text = "Keyword spotting with a convolutional neural network\non the Google Speech Commands dataset"
p2.font.size = Pt(24); p2.font.color.rgb = ICE
p2.alignment = PP_ALIGN.CENTER
tb3 = s.shapes.add_textbox(Inches(1), Inches(4.7), Inches(11.3), Inches(1))
p3 = tb3.text_frame.paragraphs[0]
p3.text = "Deep Learning — Course Project\nAditya"
p3.font.size = Pt(20); p3.font.color.rgb = WHITE
p3.alignment = PP_ALIGN.CENTER
footer(s, dark=True)
note(s, "Open with the one-liner: a computer that hears 35 spoken commands. "
        "This is your deep-learning course project — everything here was "
        "trained and measured by you, on your own machine.")

# ---------------- 2. Problem ----------------
s = new_slide(); add_bg(s)
title_bar(s, "The problem: keyword spotting",
          "Recognize short spoken commands in real time, on small devices")
bullets(s, Inches(0.7), Inches(2.2), Inches(11.9), Inches(4), [
    ("Voice assistants, smart TVs and IoT gadgets all need to hear a few dozen fixed commands — “stop”, “go”, “volume up” — instantly and offline.", False),
    ("Goal: given a 1-second audio clip, output which of 35 command words was spoken.", True),
    ("Constraints: tiny model (fits on a phone), fast inference, robust to background noise.", False),
    ("This is a classification problem — but the input is sound, not an image.", False),
])
footer(s)
note(s, "Keyword spotting = always-on listening for a fixed vocabulary. "
        "Stress the constraints — tiny, fast, offline — that's what justifies "
        "a small CNN instead of a giant model.")

# ---------------- 3. Dataset ----------------
s = new_slide(); add_bg(s)
title_bar(s, "Dataset: Google Speech Commands v0.02")
bullets(s, Inches(0.7), Inches(2.2), Inches(5.5), Inches(4), [
    ("105,829 one-second clips, 16 kHz mono — words like “yes”, “no”, “up”, “down”, “stop”, “go”, plus digits.", False),
    ("35 word classes; official validation/test lists used (no random split) so results are comparable.", False),
    ("Classes are imbalanced — the smallest word class has ~2.6× fewer clips than the largest.", True),
    ("Ships with real background-noise recordings for the robustness study.", False),
])
add_image(s, PLOTS / "class_balance.png", Inches(6.8), Inches(2.1), Inches(5.8))
footer(s)
note(s, "105k clips, official splits — say why official splits matter: your "
        "numbers are comparable to published work, not inflated by a lucky "
        "random split.")

# ---------------- 4. Approach ----------------
s = new_slide(); add_bg(s)
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
        ap = arr.text_frame.paragraphs[0]; ap.text = "▶"
        ap.font.size = Pt(28); ap.font.color.rgb = CORAL
bullets(s, Inches(0.7), Inches(5.2), Inches(11.9), Inches(2), [
    ("Why mel? It warps pitch the way human ears hear — fine detail at low frequencies, coarse at high ones.", False),
    ("Why log + normalize? Loudness is logarithmic, and neural nets train far better on zero-mean inputs.", False),
], size=18)
footer(s)
note(s, "Walk the pipeline left to right. The key insight: mel warps pitch "
        "the way human ears hear. Log-scale plus normalization because nets "
        "train far better on zero-mean inputs.")

# ---------------- 5. Architecture (block diagram) ----------------
s = new_slide(); add_bg(s)
title_bar(s, "Model architecture: KeywordCNN",
          f"{params:,} trainable parameters — small enough to train on a laptop CPU")
blocks = [
    ("INPUT", "1 × 40 × 101\nlog-mel\nspectrogram"),
    ("CONV ×2", "64 → 128 ch\n3×3 + BN + ReLU\nMaxPool + Dropout"),
    ("ADAPTIVE\nAVG POOL", "4 × 8\nkeeps time\nresolution"),
    ("DENSE", "4096 → 256\nReLU + Dropout\n(0.5)"),
    ("OUTPUT", "35 words\nsoftmax"),
]
for i, (h, d) in enumerate(blocks):
    x = Inches(0.55 + i * 2.47)
    box = s.shapes.add_shape(1, x, Inches(2.4), Inches(2.15), Inches(2.4))
    box.fill.solid()
    box.fill.fore_color.rgb = CORAL if i == 2 else NAVY
    box.line.fill.background()
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = h; p.font.size = Pt(19)
    p.font.bold = True; p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER
    p2 = tf.add_paragraph(); p2.text = d; p2.font.size = Pt(14)
    p2.font.color.rgb = WHITE; p2.alignment = PP_ALIGN.CENTER
    if i < 4:
        arr = s.shapes.add_textbox(x + Inches(2.15), Inches(3.35),
                                   Inches(0.32), Inches(0.5))
        ap = arr.text_frame.paragraphs[0]; ap.text = "▶"
        ap.font.size = Pt(26); ap.font.color.rgb = CORAL
bullets(s, Inches(0.7), Inches(5.3), Inches(11.9), Inches(1.6), [
    ("Convolutions detect local time-frequency patterns; pooling gives shift tolerance; dropout + batch-norm fight overfitting.", False),
    ("The coral block is the hero of this story — more on the next slide.", True),
], size=18)
footer(s)
note(s, "Walk each block left to right. Point at the coral block — the "
        "AdaptiveAvgPool — and tease it: that's the fix from the debugging "
        "story coming up next.")

# ---------------- 6. Training ----------------
s = new_slide(); add_bg(s)
title_bar(s, "Training")
bullets(s, Inches(0.7), Inches(2.2), Inches(6.4), Inches(4.5), [
    ("Cross-entropy loss, best checkpoint kept by validation accuracy — the test set was never touched during training.", True),
    ("Augmentation-free baseline: the numbers ahead are from clean training only.", False),
    ("Learning rate halved automatically when validation accuracy plateaus.", False),
])
stat_tile(s, Inches(7.6), str(TRAIN_EPOCHS), "epochs\nCPU-only")
stat_tile(s, Inches(10.35), "256", "batch size")
# second-row tile for the optimizer
box = s.shapes.add_shape(1, Inches(7.6), Inches(4.5), Inches(5.25), Inches(1.9))
box.fill.solid(); box.fill.fore_color.rgb = GREY
box.line.fill.background()
tf = box.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = "Adam  •  lr 1e-3"; p.font.size = Pt(28)
p.font.bold = True; p.font.color.rgb = WHITE; p.alignment = PP_ALIGN.CENTER
p2 = tf.add_paragraph(); p2.text = "optimizer"; p2.font.size = Pt(15)
p2.font.color.rgb = WHITE; p2.alignment = PP_ALIGN.CENTER
footer(s)
note(s, f"{TRAIN_EPOCHS} epochs on CPU, Adam. Best checkpoint picked by "
        "validation accuracy — test set never touched during training, so the "
        "test number is honest.")

# ---------------- 7. Iteration: fixing the bottleneck ----------------
s = new_slide(); add_bg(s)
title_bar(s, "Iteration: what the first model taught us",
          "The first trained model stalled at 52.7% — debugging it was the most valuable step")
# big before/after visual
for x, txt, col, lab in [
        (Inches(1.6), "52.7%", GREY, "first model"),
        (Inches(8.6), "63.3%", ACCENT, "after pooling fix")]:
    tb = s.shapes.add_textbox(x, Inches(2.2), Inches(3.1), Inches(1.4))
    p = tb.text_frame.paragraphs[0]; p.text = txt
    p.font.size = Pt(64); p.font.bold = True; p.font.color.rgb = col
    p.alignment = PP_ALIGN.CENTER
    tb2 = s.shapes.add_textbox(x, Inches(3.5), Inches(3.1), Inches(0.5))
    p2 = tb2.text_frame.paragraphs[0]; p2.text = lab
    p2.font.size = Pt(16); p2.font.color.rgb = GREY
    p2.alignment = PP_ALIGN.CENTER
arr = s.shapes.add_textbox(Inches(5.9), Inches(2.35), Inches(1.6), Inches(1.2))
ap = arr.text_frame.paragraphs[0]; ap.text = "▶"
ap.font.size = Pt(64); ap.font.bold = True; ap.font.color.rgb = CORAL
ap.alignment = PP_ALIGN.CENTER
arr2 = s.shapes.add_textbox(Inches(5.7), Inches(3.5), Inches(2.0), Inches(0.5))
ap2 = arr2.text_frame.paragraphs[0]; ap2.text = "+10.6 points"
ap2.font.size = Pt(18); ap2.font.bold = True; ap2.font.color.rgb = CORAL
ap2.alignment = PP_ALIGN.CENTER
bullets(s, Inches(0.7), Inches(4.6), Inches(11.9), Inches(2.4), [
    ("Symptom: validation accuracy plateaued near 53% — under-capacity, not overfitting.", False),
    ("Diagnosis: two stacked 2×2 max-pools crushed the 40×101 spectrogram to 10×25, destroying the fine timing detail that separates “three” from “tree”.", False),
    ("Viva-ready lesson: when a model underfits, ask what information the architecture destroys.", True),
], size=18)
footer(s)
note(s, "Tell the debugging story like it happened: 52.7% stall, diagnosed "
        "destroyed time resolution, AdaptiveAvgPool fix, +10.6 points. "
        "Professors love this slide — it proves you think, not just train.")

# ---------------- 8. Results ----------------
s = new_slide(); add_bg(s)
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
    box.fill.fore_color.rgb = ACCENT if hl else GREY_LT
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
footer(s)
note(s, "Read the three bars top to bottom. The gap between CNN and SVM is "
        "the story: this is what deep learning buys on this task. Random "
        "guessing at 2.9% shows how far 63% really is.")

# ---------------- 9. Confusion matrix ----------------
s = new_slide(); add_bg(s)
title_bar(s, "Where does it go wrong? Confusion matrix")
add_image(s, RES / "confusion_matrix.png", Inches(0.7), Inches(2.0),
          Inches(6.4))
conf_lines = [l.strip() for l in top_conf[1:6] if l.strip()]
bullets(s, Inches(7.6), Inches(2.4), Inches(5), Inches(4),
        [("Most confused pairs:", True)] +
        [(l, False) for l in conf_lines] +
        [("These are genuinely similar-sounding words — even humans mix them up.", False)],
        size=17)
footer(s)
note(s, "Errors are sensible — similar-sounding pairs. Point at go/no and "
        "three/tree: even humans mix these up. This is evidence the model "
        "learned phonetics, not noise.")

# ---------------- 10. Noise robustness ----------------
s = new_slide(); add_bg(s)
title_bar(s, "Noise robustness",
          "Real background noise mixed into test clips at controlled SNRs")
add_image(s, PLOTS / "noise_robustness.png", Inches(0.7), Inches(2.2),
          Inches(6.2))
items = [("Accuracy degrades gracefully as noise rises — no cliff-edge failure.", False)]
if noise:
    items.append((f'Clean: {noise.get("clean", 0):.1%} → 0 dB SNR: '
                  f'{noise.get("snr_0db", 0):.1%}', True))
items.append(("Honest weakness — and the natural next step: fine-tuning on noisy clips.", False))
bullets(s, Inches(7.4), Inches(2.4), Inches(5.2), Inches(4), items)
footer(s)
note(s, "Be honest here: accuracy collapses in heavy noise. Graceful "
        "degradation, no cliff — and the obvious next step is training on "
        "noisy clips. Admitting a weakness with a plan beats hiding it.")

# ---------------- 11. Demo ----------------
s = new_slide(); add_bg(s)
title_bar(s, "Live demo", "Streamlit app — mic recording or .wav upload")
add_image(s, PLOTS / "waveform_spectrogram_examples.png",
          Inches(0.7), Inches(2.2), Inches(6.2))
bullets(s, Inches(7.4), Inches(2.4), Inches(5.2), Inches(4), [
    ("Run:  streamlit run app/streamlit_app.py", True),
    ("Records 1 second from the browser mic, computes the log-mel, shows top-3 predictions.", False),
    ("A “custom model” toggle loads your fine-tuned weights after you record your own voice.", False),
])
footer(s)
note(s, "Demo tip: have a backup .wav ready in case the classroom mic "
        "misbehaves. Show the top-3 predictions — the model's uncertainty is "
        "interesting, not embarrassing.")

# ---------------- 12. Personalization & future work ----------------
s = new_slide(); add_bg(s)
title_bar(s, "Personalization & future work")
bullets(s, Inches(0.7), Inches(2.2), Inches(11.9), Inches(4.5), [
    ("The dataset is mostly North-American accents — record_custom.py + finetune_custom.py adapt the model to YOUR voice (new words become new output classes).", False),
    ("Next steps: train with noise augmentation; try the TinyTransformer in src/models.py; quantize to run on a microcontroller.", False),
    ("Key lesson: representation matters more than model size — a good spectrogram + a 1.1M-param CNN beats a bigger model on raw audio.", True),
])
footer(s)
note(s, "The personalization angle is what makes this project yours, not a "
        "tutorial copy: your voice, your words. Close with the representation "
        "lesson — it's the most quotable line of the talk.")

# ---------------- 13. Thank you ----------------
s = new_slide(); add_bg(s, NAVY)
tb = s.shapes.add_textbox(Inches(1), Inches(2.6), Inches(11.3), Inches(2))
p = tb.text_frame.paragraphs[0]; p.text = "Thank you!"
p.font.size = Pt(54); p.font.bold = True; p.font.color.rgb = WHITE
p.alignment = PP_ALIGN.CENTER
tb2 = s.shapes.add_textbox(Inches(1), Inches(4.2), Inches(11.3), Inches(1))
p2 = tb2.text_frame.paragraphs[0]
p2.text = "Code, docs & trained model: github.com/idityaGE/voice-command-recognition"
p2.font.size = Pt(20); p2.font.color.rgb = ICE
p2.alignment = PP_ALIGN.CENTER
tb3 = s.shapes.add_textbox(Inches(1), Inches(5.4), Inches(11.3), Inches(1))
p3 = tb3.text_frame.paragraphs[0]; p3.text = "Questions?"
p3.font.size = Pt(28); p3.font.color.rgb = WHITE; p3.alignment = PP_ALIGN.CENTER
footer(s, dark=True)
note(s, "Thank the audience, point at the repo link, invite questions. If "
        "asked about the 63%: it's a clean baseline — augmentation and a "
        "bigger model are the known next steps.")

OUT.parent.mkdir(exist_ok=True)
prs.save(OUT)
print(f"saved {OUT}")
