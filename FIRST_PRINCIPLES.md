# Voice Command Recognition — From First Principles

This document explains *why* every step of the project exists, starting from
physics. If you understand this file, you can explain the whole project in a
viva without memorizing anything.

---

## 1. Sound is wiggles in air

When you say "stop", your vocal cords and mouth shape the air into a pressure
wave that changes over time. A microphone turns that pressure wave into an
electrical signal — a single number going up and down, thousands of times per
second.

## 2. Sampling: freezing the wiggle into numbers

A computer can't store a continuous wave, so it *samples* it: it measures the
air pressure at evenly spaced instants. Our dataset uses **16,000 samples per
second (16 kHz)**. Each 1-second clip is therefore just a list of 16,000
numbers — this is the "waveform" you see in `results/plots/`.

Why 16 kHz and not more? The **Nyquist theorem** says: to capture a frequency
*f*, you must sample at more than *2f*. Human speech carries almost all its
information below 8 kHz, so 16 kHz is enough — and half the data of 32 kHz.

## 3. The problem with the raw waveform

16,000 numbers *do* contain the word "stop", but in an awkward form:

- Say "stop" slightly faster or slower and all 16,000 numbers shift.
- Two recordings of "stop" look completely different as number lists, even
  though they *sound* the same.
- What matters for recognizing a word is **which frequencies are present at
  which moment** (the "sss" hiss, the "t" burst, the vowel's hum) — not the
  exact wiggle shape.

So we transform the waveform into a representation that exposes
frequency-over-time. That transform is the **spectrogram**.

## 4. The spectrogram: slicing time, measuring pitch

Take a short slice of the waveform (25 milliseconds — short enough that the
sound inside it is roughly steady) and ask: *how much of each pitch is in
this slice?* The **Fourier transform** answers exactly that: it decomposes
any wiggle into a sum of pure tones and tells you each tone's strength.

Then slide the 25 ms window forward by 10 ms and repeat. Stack the results
side by side and you get a 2D image:

- **horizontal axis = time** (~101 slices for 1 second)
- **vertical axis = pitch/frequency**
- **brightness = how loud that pitch is at that moment**

A spoken word now *looks* like a picture: "sheila" shows a bright noisy band
(the "sh"), "zero" shows smooth vowel stripes. Two utterances of the same
word produce similar pictures even if the raw wiggles differ — this is why
the spectrogram is the right input.

## 5. The mel scale: hearing like a human

Humans don't hear pitch linearly: we easily tell 200 Hz from 300 Hz, but
barely distinguish 7000 Hz from 7100 Hz. The **mel scale** warps the
frequency axis to match human hearing — fine resolution at low pitches,
coarse at high ones. Applying 40 triangular mel filters to each slice gives
the **mel spectrogram** (40 rows × ~101 columns).

Two final touches used in `src/features.py`:

- **Log scale (decibels).** Loudness perception is logarithmic — a whisper
  to a shout is a 1000× energy change but doesn't *feel* 1000× louder. The
  log compresses the range so quiet details survive.
- **Per-clip normalization.** Subtract the mean, divide by the standard
  deviation. Neural networks train far better when inputs are roughly
  zero-mean, unit-variance — otherwise loud clips dominate learning.

## 6. The CNN: learning to see words

A **convolutional neural network** treats the log-mel spectrogram as an
image and learns visual patterns in it:

- A **convolution** slides a small filter (3×3) over the image and fires
  where it finds a matching pattern — e.g. a vertical edge could be a sudden
  consonant burst, a horizontal stripe a steady vowel. The first layer learns
  64 such pattern-detectors; the second layer (128 filters) combines them
  into bigger shapes ("hiss followed by vowel").
- **ReLU** (`max(0, x)`) keeps only positive firings — it gives the network
  its ability to model non-linear, interesting functions. Without it, the
  whole network would collapse into one linear operation.
- **Max-pooling** (2×2) keeps the strongest firing in each little region and
  throws away the rest: this makes the network tolerant to small shifts in
  time/pitch and shrinks the image so later layers are cheap.
- **Batch normalization** re-centers each layer's outputs during training,
  which keeps gradients healthy and lets us use a larger learning rate.
- **Dropout** randomly switches off neurons during training, forcing the
  network to not rely on any single pattern — a cheap, effective cure for
  memorization (overfitting).
- After two conv blocks, **adaptive average pooling** squeezes each of the
  128 feature maps down to a 4×8 summary (4096 numbers total — deliberately
  keeping time resolution, which is what distinguishes confusable words like
  "four" vs "forward"), and a small **dense head** (4096 → 256 → 35) votes on
  which of the 35 words it is.

Total: ~1.1 million learnable numbers (parameters) — still small enough to
train on a laptop CPU in a couple of hours.

## 7. Softmax + cross-entropy: turning scores into learning

The network's final layer outputs 35 raw scores ("logits") — one per word.
**Softmax** squashes them into probabilities that sum to 1:
"87% yes, 4% no, …". During training we know the true word, so we penalize
the network with **cross-entropy loss**: essentially `-log(probability it
gave the right word)`. Confident-and-right → tiny loss; confident-and-wrong
→ huge loss. That single number is the compass for all learning.

## 8. Training: walking downhill in 1.1 million dimensions

1. Show the network a batch of 256 clips, get its probabilities.
2. Compute the average cross-entropy loss.
3. **Backpropagation** (the chain rule from calculus, applied layer by
   layer) computes how much each of the ~1.1M parameters contributed to the
   loss — i.e. which direction to nudge each one to reduce it.
4. **Adam** (the optimizer) nudges every parameter a small step downhill,
   adapting the step size per parameter based on past gradients.
5. Repeat for ~84,000 training clips × 15 epochs. Watch validation accuracy
   (on clips the network never trains on) — when it stops improving, stop.

The learning rate is reduced automatically when validation accuracy plateaus
(`ReduceLROnPlateau`): big steps early for speed, small steps late for
precision.

## 9. Evaluation: trust, but verify

- We evaluate on the **official test split** — clips the network never saw
  in training *or* validation. Test accuracy is the honest number.
- The **confusion matrix** (35×35) shows *which* words get mixed up
  ("three"↔"tree" — they genuinely sound alike). The most confused pairs
  are listed in `results/top_confusions.txt`.
- **Per-class precision/recall** in `results/metrics.json` reveals weak
  spots: a word with low recall is often under-represented in training
  (see the class-balance plot — the smallest class has ~2.6× fewer clips
  than the largest).

## 10. The experiments around the model

- **MFCC + SVM baseline** (`src/baseline_svm.py`): before deep learning,
  speech was classified with hand-designed features (MFCCs ≈ a coarser
  cousin of mel spectrograms) and a support-vector machine. Our baseline
  scores lower than the CNN — which is exactly the point: it quantifies
  *what the deep network adds*.
- **Noise robustness** (`src/noise_experiment.py`): real microphones hear
  fans, traffic, chatter. We mix genuine background noise into test clips
  at controlled SNRs and watch accuracy decay — this tells you how the
  model would behave outside the lab.
- **Your own voice** (`record_custom.py` + `finetune_custom.py`): the
  dataset is mostly American accents. Recording your own clips and
  fine-tuning (small learning rate, replaying old data so it doesn't
  forget) personalizes the model — the same idea behind on-device
  personalization in phones.

---

**One-sentence summary for the viva:** *"We convert each 1-second voice clip
into a log-mel spectrogram — a time-frequency image matched to human
hearing — and train a small convolutional network to recognize visual
patterns in it, reaching ~90% accuracy on 35 command words, with experiments
showing how it compares to a classical baseline and how it degrades in
noise."*
