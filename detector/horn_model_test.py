"""Score a recording with the horn sound model:  python horn_model_test.py horn.wav [models/yamnet.onnx] [models/horn_head.npz]"""
import os
import sys
import wave

import numpy as np

from detector.horn_classifier import HornClassifier

if len(sys.argv) < 2:
    sys.exit(__doc__)
model = sys.argv[2] if len(sys.argv) > 2 else "models/yamnet.onnx"
head = sys.argv[3] if len(sys.argv) > 3 else "models/horn_head.npz"
try:
    with wave.open(sys.argv[1], "rb") as w:
        sr, width, ch = w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    data = np.frombuffer(raw, dtype={1: np.uint8, 2: np.int16, 4: np.int32}[width]).astype(np.float32)
    data = data.reshape(-1, ch)[:, 0]
    data = (data - 128) / 128 if width == 1 else data / float(2 ** (8 * width - 1))
except Exception:
    import soundfile as sf
    data, sr = sf.read(sys.argv[1], dtype="float32", always_2d=True)
    data = data[:, 0]

clf = HornClassifier(model, head_path=head if os.path.exists(head) else None)
thr = clf.default_min_score or 0.3
v = clf.judge(data, sr)
print("heard:", v.note())
print(f"horn {v.horn:.2f}   voice {v.voice:.2f}   alarm/siren {v.alarm:.2f}   (min {thr:.2f}) ->",
      "HORN" if v.accepted(thr) else "not a horn")
