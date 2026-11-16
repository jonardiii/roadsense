"""Second opinion on a horn: a pre-trained sound classifier (YAMNet, trained on AudioSet).

The frequency check finds sounds that look like a horn. YAMNet knows what 521 kinds of sound *are*, among them
"Vehicle horn, car horn, honking", "Shout", "Yell" and "Speech", so it can tell a horn from a person shouting.
It is used as a filter: a horn event is kept only if YAMNet gives the horn classes at least `min_score`.

It needs the model file as ONNX (see README, "Horn classifier") and onnxruntime, which the plate reader already installs.
"""
import csv
import logging
import math
import os
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("roadsense.horn_classifier")

SAMPLE_RATE = 16000
HORN_CLASSES = (302, 303, 312)         # Vehicle horn / car horn / honking, Toot, Air horn / truck horn
# People: Speech, Shout, Bellow, Yell, Children shouting, Screaming, Hubbub
VOICE_CLASSES = (0, 6, 7, 9, 10, 11, 65)
# Sounds that are tonal or alarm-like and are easy to confuse with a horn but are not a driver honking:
# car alarm, alarm, sirens, reversing beeps, beep, buzzer, whistles
ALARM_CLASSES = (304, 313, 317, 318, 319, 382, 390, 391, 392, 396, 397, 475)
N_CLASSES = 521
MIN_SAMPLES = 15600                    # YAMNet looks at 0.975 s at a time


def _load_names():
    path = os.path.join(os.path.dirname(__file__), "yamnet_class_map.csv")
    try:
        with open(path, encoding="utf-8") as fh:
            return {int(r[0]): r[2] for r in list(csv.reader(fh))[1:] if len(r) >= 3}
    except (OSError, ValueError):
        return {}


@dataclass
class Judgement:
    horn: float                  # best score of the horn classes
    voice: float                 # best score of people sounds (speech, shouting, screaming)
    alarm: float                 # best score of sirens, alarms, beeps and whistles
    top: list = field(default_factory=list)    # the three strongest sounds heard: [(name, score), ...]

    def accepted(self, min_score):
        """A horn only if the model is sure enough, and no voice or alarm is stronger than the horn."""
        return self.horn >= min_score and self.horn >= max(self.voice, self.alarm)

    def note(self):
        return "; ".join(f"{n} {s:.2f}" for n, s in self.top)[:200]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit_features(scores):
    p = np.clip(np.asarray(scores, dtype=np.float64), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class HornClassifier:
    def __init__(self, model_path, head_path=None, normalize=False):
        import onnxruntime as ort
        self.normalize = normalize
        self.head = None
        self.default_min_score = None
        if head_path and os.path.exists(head_path):
            h = np.load(head_path)
            self.head = {k: h[k] for k in ("w", "b", "mean", "std")}
            self.default_min_score = float(h["threshold"])
            self.normalize = bool(h["normalize"])
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.rank = len(inp.shape)
        last = inp.shape[-1]
        self.fixed = last if isinstance(last, int) and last > 0 else None
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.names = _load_names()

    def _windows(self, wave):
        size = self.fixed or max(MIN_SAMPLES, len(wave))
        if len(wave) < size:                      # short sound: centre it in silence
            pad = size - len(wave)
            wave = np.pad(wave, (pad // 2, pad - pad // 2))
        if self.fixed is None or len(wave) == size:
            return [wave[:size]] if self.fixed else [wave]
        hop = size // 2
        return [wave[i:i + size] for i in range(0, len(wave) - size + 1, hop)] or [wave[-size:]]

    def scores(self, samples, sample_rate):
        """Class scores for every 0.975 s of the sound, shape (frames, 521)."""
        wave = np.asarray(samples, dtype=np.float32).reshape(-1)
        if sample_rate != SAMPLE_RATE:
            from scipy.signal import resample_poly
            g = math.gcd(int(sample_rate), SAMPLE_RATE)
            wave = resample_poly(wave, SAMPLE_RATE // g, int(sample_rate) // g).astype(np.float32)
        rows = []
        for win in self._windows(wave):
            x = win[None, :] if self.rank == 2 else win
            for arr in self.sess.run(self.out_names, {self.input_name: x.astype(np.float32)}):
                arr = np.asarray(arr)
                if arr.ndim >= 1 and arr.shape[-1] == N_CLASSES:
                    arr = arr.reshape(-1, N_CLASSES)
                    if arr.min() < 0.0 or arr.max() > 1.0:   # logits, not scores
                        arr = _sigmoid(arr)
                    rows.append(arr)
                    break
        if not rows:
            raise RuntimeError("The horn model has no 521-class output. Is it the YAMNet model?")
        return np.concatenate(rows, axis=0)

    def features(self, samples, sample_rate):
        """The 521 sound-class scores of a clip (the strongest moment of each), as logits: input of the learned layer."""
        wave = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self.normalize and len(wave) and np.abs(wave).max() > 1e-4:
            wave = wave * (0.9 / np.abs(wave).max())      # quiet recordings are brought to a normal level
        best = self.scores(wave, sample_rate).max(axis=0)
        return best

    def judge(self, samples, sample_rate):
        """What the sound is: a Judgement with the horn, voice and alarm scores and the three strongest sounds."""
        best = self.features(samples, sample_rate)    # strongest moment of each sound class
        top = [(self.names.get(int(i), f"class {int(i)}"), float(best[i])) for i in np.argsort(best)[::-1][:3]]
        if self.head is not None:                      # the layer trained on horn recordings decides
            x = logit_features(best)
            z = float(((x - self.head["mean"]) / self.head["std"]) @ self.head["w"] + self.head["b"])
            return Judgement(horn=float(_sigmoid(z)), voice=0.0, alarm=0.0, top=top)
        return Judgement(horn=float(best[list(HORN_CLASSES)].max()), voice=float(best[list(VOICE_CLASSES)].max()),
                         alarm=float(best[list(ALARM_CLASSES)].max()), top=top)
