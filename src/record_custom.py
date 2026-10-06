"""Record your own 1-second voice clips for custom words.

Saves mono 16 kHz wav files to data/custom/<word>/clip_000.wav, ...

Usage:
    python record_custom.py --word stop --n 20
    python record_custom.py --word "ruk jao" --n 20   # multi-word commands ok

Tip: record in a quiet room, ~15 cm from the mic, and vary your tone a little
between clips -- it makes the fine-tuned model much more robust.

Requires: pip install sounddevice  (bundles PortAudio; works on Windows)
"""

import argparse
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", required=True, help="word/command being recorded")
    ap.add_argument("--n", type=int, default=20, help="how many clips to record")
    ap.add_argument("--out", default="../data/custom")
    ap.add_argument("--seconds", type=float, default=1.0)
    args = ap.parse_args()

    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        raise SystemExit("pip install sounddevice soundfile  (then re-run)")

    sr = 16000
    word_dir = Path(args.out) / args.word.strip().replace(" ", "_")
    word_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(word_dir.glob("*.wav")))
    print(f"recording {args.n} clips of \"{args.word}\" -> {word_dir}")
    print("speak right AFTER the beep. press Ctrl+C to stop early.\n")

    for i in range(args.n):
        print(f"[{i+1}/{args.n}] get ready...", end=" ", flush=True)
        time.sleep(0.8)
        print("BEEP - speak now!")
        sd.play(np.sin(2 * np.pi * 880 * np.arange(int(sr * 0.15)) / sr), sr)
        sd.wait()
        clip = sd.rec(int(sr * args.seconds), samplerate=sr,
                      channels=1, dtype="float32")
        sd.wait()
        path = word_dir / f"clip_{existing + i:03d}.wav"
        sf.write(path, clip, sr)
        print(f"  saved {path.name}")

    print(f"\ndone. {args.n} clips in {word_dir}")
    print("next: python finetune_custom.py --data ../data "
          "--custom-dir ../data/custom --ckpt ../models/keyword_cnn.pt")


if __name__ == "__main__":
    main()
