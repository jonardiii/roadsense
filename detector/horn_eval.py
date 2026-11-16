"""Measure how well the horn detection tells a car horn from other sounds, on a public sound dataset.

   python horn_eval.py --dataset urbansound8k --root data\\UrbanSound8K
   python horn_eval.py --dataset esc50        --root data\\ESC-50-master
   python horn_eval.py --dataset hornbase     --root data\\HornBase        (extract Dataset.zip inside it first)
   python horn_eval.py --dataset folders      --root data\\my_sounds     (one folder per sound; "horn" = the car horns)

For every clip it asks two detectors whether it is a car horn:
  1. the frequency check alone (what the detector did before the sound model), and
  2. the frequency check plus the pre-trained sound model (YAMNet), the way the program works now.
It prints, per kind of sound, how many clips were accepted as a horn, then the overall precision and recall,
and a table for different values of horn_model_min_score. --csv saves every clip's result.

Datasets (free): UrbanSound8K https://urbansounddataset.weebly.com/urbansound8k.html (class car_horn)
                 ESC-50        https://github.com/karolpiczak/ESC-50 (category car_horn)
Only use clips of one dataset at a time. Clips are tested one by one, the way a horn event is handed to the model.
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np

from detector.audio import HornDetector
from detector.horn_classifier import SAMPLE_RATE, HornClassifier

HORN_NAMES = {"car_horn", "horn", "car horn"}


def find_file(root, name):
    """The folder with a file called `name` somewhere under root (the zip often has an extra folder level)."""
    for folder, _dirs, files in os.walk(root):
        for f in files:
            if f.lower() == name.lower():
                return os.path.join(folder, f)
    sys.exit(f"Could not find {name} under {root}. Check --root points at the extracted dataset folder.")


def list_clips(dataset, root):
    """[(path, kind of sound)]"""
    clips = []
    if dataset == "urbansound8k":
        meta = find_file(root, "UrbanSound8K.csv")
        base = os.path.join(os.path.dirname(os.path.dirname(meta)), "audio")
        with open(meta, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                clips.append((os.path.join(base, f"fold{r['fold']}", r["slice_file_name"]), r["class"]))
    elif dataset == "esc50":
        meta = find_file(root, "esc50.csv")
        base = os.path.join(os.path.dirname(os.path.dirname(meta)), "audio")
        with open(meta, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                clips.append((os.path.join(base, r["filename"]), r["category"]))
    elif dataset == "hornbase":
        meta = find_file(root, "hornbase.csv")
        wavs = {}
        for folder, _dirs, files in os.walk(root):
            for f in files:
                if f.lower().endswith(".wav"):
                    wavs[f] = os.path.join(folder, f)
        with open(meta, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["file"] in wavs:
                    clips.append((wavs[r["file"]], r["class"].strip()))
        if not clips:
            sys.exit("No HornBase clips found. Extract Dataset.zip inside the HornBase folder first.")
    else:
        for kind in sorted(os.listdir(root)):
            folder = os.path.join(root, kind)
            if os.path.isdir(folder):
                for f in sorted(os.listdir(folder)):
                    if f.lower().endswith((".wav", ".flac", ".ogg")):
                        clips.append((os.path.join(folder, f), kind))
    return clips


def load(path):
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(sr, SAMPLE_RATE)
        data = resample_poly(data, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return data


def frequency_check(wave, cfg):
    """Would the frequency check alone call it a horn? The clip is placed after 3 s of quiet background."""
    det = HornDetector(SAMPLE_RATE, min_excess_db=cfg["excess"], min_tonal=cfg["tonal"],
                       min_similarity=cfg["similarity"], min_duration_s=cfg["duration"], cooldown_s=2.0)
    rng = np.random.default_rng(0)
    background = (rng.standard_normal(3 * SAMPLE_RATE) * 0.002).astype(np.float32)
    audio = np.concatenate([background, wave, np.zeros(SAMPLE_RATE, np.float32)])
    t0, events = 1000.0, []
    for i in range(0, len(audio), 512):
        events += det.feed(audio[i:i + 512], t0 + i / SAMPLE_RATE)
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["urbansound8k", "esc50", "hornbase", "folders"])
    ap.add_argument("--root")
    ap.add_argument("--from-csv", default="", help="re-score a results file saved earlier (no audio or model needed)")
    ap.add_argument("--model", default="models/yamnet.onnx")
    ap.add_argument("--min-score", type=float, default=None, help="default 0.3, or the threshold saved with the head")
    ap.add_argument("--head", default="", help="a horn_head.npz made by horn_train_head.py (test the fine-tuned detector)")
    ap.add_argument("--normalize", action="store_true", help="bring every clip to the same loudness before the model")
    ap.add_argument("--max-per-class", type=int, default=0, help="test at most this many clips of each sound (0 = all)")
    ap.add_argument("--csv", default="", help="save the result of every clip here")
    args = ap.parse_args()
    rows = []
    if args.min_score is None and args.from_csv:
        args.min_score = 0.3
    if args.from_csv:
        with open(args.from_csv, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows.append({"file": r["file"], "kind": r["kind"], "is_horn": r["is_horn"] == "True",
                             "freq": r["freq"] == "True", "horn": float(r["horn"]), "voice": float(r["voice"]),
                             "alarm": float(r["alarm"]), "heard": r.get("heard", "")})
    else:
        if not args.dataset or not args.root:
            sys.exit("Give --dataset and --root (or --from-csv).")
        if not os.path.exists(args.model):
            sys.exit(f"No sound model at {args.model}. See the README, 'Horn classifier'.")
        clf = HornClassifier(args.model, head_path=args.head or None, normalize=args.normalize)
        if args.min_score is None:
            args.min_score = clf.default_min_score or 0.3
        cfg = {"excess": 10.0, "tonal": 0.6, "similarity": 0.85, "duration": 0.2}
        clips = list_clips(args.dataset, args.root)
        if args.max_per_class:
            seen = collections.Counter()
            kept = []
            for path, kind in clips:
                if seen[kind] < args.max_per_class:
                    seen[kind] += 1
                    kept.append((path, kind))
            clips = kept
        rows = []
        for n, (path, kind) in enumerate(clips, 1):
            try:
                wave = load(path)
                freq = bool(frequency_check(wave, cfg))
                v = clf.judge(wave, SAMPLE_RATE)
            except Exception as exc:  # noqa: BLE001
                print(f"skipped {path}: {exc}")
                continue
            rows.append({"file": os.path.basename(path), "kind": kind, "is_horn": kind.lower() in HORN_NAMES,
                         "freq": freq, "horn": v.horn, "voice": v.voice, "alarm": v.alarm, "heard": v.note()})
            if n % 100 == 0:
                print(f"  {n}/{len(clips)} clips", flush=True)
    if not rows:
        sys.exit("No clips were tested. Check --root.")

    def model_ok(r, thr):
        return r["horn"] >= thr and r["horn"] >= max(r["voice"], r["alarm"])

    def verdict(r, thr):          # how the detector works now: the frequency check first, then the sound model
        return r["freq"] and model_ok(r, thr)

    print(f"\n{'sound':22} {'clips':>6} {'frequency only':>18} {'sound model only':>20} {'both (now)':>16}")
    for kind in sorted({r["kind"] for r in rows}, key=lambda k: (k.lower() not in HORN_NAMES, k)):
        sub = [r for r in rows if r["kind"] == kind]
        a = sum(r["freq"] for r in sub)
        m = sum(model_ok(r, args.min_score) for r in sub)
        b = sum(verdict(r, args.min_score) for r in sub)
        star = "  <- the horns" if kind.lower() in HORN_NAMES else ""
        print(f"{kind:22} {len(sub):6d} {a:9d} ({a / len(sub):4.0%}) {m:11d} ({m / len(sub):4.0%}) {b:8d} ({b / len(sub):4.0%}){star}")

    def stats(pred):
        tp = sum(1 for r in rows if r["is_horn"] and pred(r))
        fn = sum(1 for r in rows if r["is_horn"] and not pred(r))
        fp = sum(1 for r in rows if not r["is_horn"] and pred(r))
        tn = sum(1 for r in rows if not r["is_horn"] and not pred(r))
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        return tp, fn, fp, tn, prec, rec

    print("\n                                  horns found  horns missed  false alarms  correct rejects  precision  recall")
    for name, pred in (("frequency check only", lambda r: r["freq"]),
                       (f"sound model only (>= {args.min_score:g})", lambda r: model_ok(r, args.min_score)),
                       ("both: frequency, then model", lambda r: verdict(r, args.min_score)),
                       ("either one (frequency or model)", lambda r: r["freq"] or model_ok(r, args.min_score))):
        tp, fn, fp, tn, p, rc = stats(pred)
        print(f"{name:32} {tp:12d} {fn:13d} {fp:13d} {tn:16d} {p:10.2f} {rc:7.2f}")
    print("\nSound model only, at other values of horn_model_min_score:")
    print("  min score   horns found   false alarms   precision   recall")
    for thr in (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        tp, fn, fp, tn, p, rc = stats(lambda r, t=thr: model_ok(r, t))
        print(f"  {thr:9.2f} {tp:13d} {fp:14d} {p:11.2f} {rc:8.2f}")
    if args.csv and not args.from_csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nEvery clip's result was saved to {args.csv}")


if __name__ == "__main__":
    main()
