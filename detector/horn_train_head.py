"""Teach the horn detector what the horns in YOUR recordings sound like (fine-tuning the sound model).

   python horn_train_head.py --dataset hornbase --root data\\HornBase
   python horn_train_head.py --dataset folders  --root data\\my_sounds      (folders: horn, and others)

The pre-trained model (YAMNet) already turns a sound into 521 scores. This trains a small layer on those scores
with your labelled clips, so quiet or muddy horns that the pre-trained model rates low are still recognised.
The clips are split into training and testing clips (HornBase has its own split files). It prints the accuracy on the
testing clips only, saves models\\horn_head.npz, and the detector uses it automatically when the file is there.
To add your own recordings from the entrance, put them in a folder named horn and a folder named other, and run
with --dataset folders (you can combine: repeat --extra folder).
"""
import argparse
import csv
import os
import random
import sys

import numpy as np

from detector.horn_classifier import SAMPLE_RATE, HornClassifier, logit_features
from horn_eval import HORN_NAMES, find_file, list_clips, load


def fit_head(X, y, l2=100.0):
    """Logistic regression on standardised features: returns w, b, mean, std."""
    from scipy.optimize import minimize
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-3
    Z = (X - mean) / std
    pos = max(1.0, y.sum())
    neg = max(1.0, len(y) - y.sum())
    sw = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))      # horns and others count equally

    def loss(params):
        w, b = params[:-1], params[-1]
        z = Z @ w + b
        # numerically stable log(1 + exp(z)) - y*z, weighted
        ll = np.logaddexp(0, z) - y * z
        grad_z = (1 / (1 + np.exp(-z)) - y) * sw
        return (sw * ll).sum() / len(y) + 0.5 * l2 * (w @ w) / len(y), \
            np.concatenate([Z.T @ grad_z / len(y) + l2 * w / len(y), [grad_z.sum() / len(y)]])

    res = minimize(loss, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x[:-1], res.x[-1], mean, std


def predict(X, head):
    w, b, mean, std = head
    return 1 / (1 + np.exp(-(((X - mean) / std) @ w + b)))


def metrics(p, y, thr):
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return tp, fp, fn, tp / (tp + fp) if tp + fp else float("nan"), tp / (tp + fn) if tp + fn else float("nan")


def split_of(path, source, hornbase_split, rng):
    """train or test. The datasets' own folds keep clips of one recording on the same side."""
    base = os.path.basename(path)
    if source == "urbansound8k":      # folder fold1..fold10: fold 10 is for testing
        return "test" if os.path.basename(os.path.dirname(path)).lower() == "fold10" else "train"
    if source == "esc50":             # file names start with the fold 1..5: fold 5 is for testing
        return "test" if base.split("-")[0] == "5" else "train"
    if source == "hornbase" and base in hornbase_split:
        return hornbase_split[base]
    return "test" if rng.random() < 0.2 else "train"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["hornbase", "folders"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--extra", action="append", default=[], help="another folder with horn/ and other/ subfolders")
    ap.add_argument("--also", action="append", default=[], metavar="NAME:FOLDER",
                    help="train on another dataset too, e.g. --also esc50:data\\ESC-50-master or "
                         "--also urbansound8k:data\\UrbanSound8K (repeat for both)")
    ap.add_argument("--max-per-class", type=int, default=0, help="at most this many clips of each sound from --also sets")
    ap.add_argument("--model", default="models/yamnet.onnx")
    ap.add_argument("--out", default="models/horn_head.npz")
    ap.add_argument("--normalize", action="store_true", help="bring every clip to the same loudness first")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="use this threshold instead of choosing one (e.g. 0.9 for fewer false alarms)")
    ap.add_argument("--precision", type=float, default=0.95, help="the detector's threshold is set for this precision")
    ap.add_argument("--l2", type=float, default=100.0, help="higher = a simpler layer that overfits less")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not os.path.exists(args.model):
        sys.exit(f"No sound model at {args.model}.")
    clf = HornClassifier(args.model, normalize=args.normalize)

    clips = [(p, k, args.dataset) for p, k in list_clips(args.dataset, args.root)]
    for extra in args.extra:
        clips += [(p, k, "folders") for p, k in list_clips("folders", extra)]
    for spec in args.also:
        name, _, folder = spec.partition(":")
        if name not in ("urbansound8k", "esc50", "hornbase") or not folder:
            sys.exit(f"--also needs NAME:FOLDER with NAME urbansound8k, esc50 or hornbase, not {spec!r}")
        got = list_clips(name, folder)
        if args.max_per_class:
            seen, kept = {}, []
            for p, k in got:
                if seen.get(k, 0) < args.max_per_class:
                    seen[k] = seen.get(k, 0) + 1
                    kept.append((p, k))
            got = kept
        clips += [(p, k, name) for p, k in got]
    split = {}
    if args.dataset == "hornbase":
        for name in ("train", "test"):
            try:
                with open(find_file(args.root, f"hornbase_{name}.csv"), encoding="utf-8") as fh:
                    for r in csv.DictReader(fh):
                        split[r["file"]] = name
            except SystemExit:
                pass
    rng = random.Random(args.seed)
    feats, labels, parts, sources = [], [], [], []
    for n, (path, kind, source) in enumerate(clips, 1):
        try:
            x = logit_features(clf.features(load(path), SAMPLE_RATE))
        except Exception as exc:  # noqa: BLE001
            print(f"skipped {path}: {exc}")
            continue
        feats.append(x)
        labels.append(1 if kind.lower() in HORN_NAMES else 0)
        parts.append(split_of(path, source, split, rng))
        sources.append(source)
        if n % 100 == 0:
            print(f"  {n}/{len(clips)} clips", flush=True)
    X, y, parts = np.array(feats), np.array(labels, dtype=np.float64), np.array(parts)
    sources = np.array(sources)
    tr, te = parts == "train", parts == "test"
    if not tr.any() or not te.any() or y[tr].sum() == 0:
        sys.exit("Need training and testing clips with horns in them.")
    print(f"\n{int(tr.sum())} training clips, {int(te.sum())} testing clips "
          f"({int(y[tr].sum())} / {int(y[te].sum())} horns)")
    head = fit_head(X[tr], y[tr], l2=args.l2)
    pte = predict(X[te], head)
    # the threshold is chosen on predictions for training clips that the layer did not train on (5-fold)
    Xtr, ytr = X[tr], y[tr]
    folds = np.arange(len(ytr)) % 5
    np.random.default_rng(args.seed).shuffle(folds)
    ptr = np.zeros(len(ytr))
    for k in range(5):
        h = fit_head(Xtr[folds != k], ytr[folds != k], l2=args.l2)
        ptr[folds == k] = predict(Xtr[folds == k], h)
    print("\nTesting clips (never used for training):")
    print("  threshold   horns found   false alarms   precision   recall")
    best_thr, best_rec = 0.5, -1.0
    for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        tp, fp, fn, pr, rc = metrics(pte, y[te], thr)
        print(f"  {thr:9.1f} {tp:13d} {fp:14d} {pr:11.2f} {rc:8.2f}")
    for thr in np.arange(0.05, 0.96, 0.01):                 # the lowest threshold that keeps the precision
        tp, fp, fn, pr, rc = metrics(ptr, ytr, thr)
        if tp + fp and pr >= args.precision and rc > best_rec:
            best_thr, best_rec = float(thr), rc
    if args.threshold:
        best_thr = args.threshold
    tp, fp, fn, pr, rc = metrics(pte, y[te], best_thr)
    print(f"\nChosen threshold {best_thr:.2f} (precision >= {args.precision:.0%} on held-out training clips): "
          f"on the testing clips {tp} horns found, {fn} missed, {fp} false alarms, precision {pr:.2f}, recall {rc:.2f}")
    if len(set(sources)) > 1:
        print("\nTesting clips by dataset at that threshold:")
        for src in sorted(set(sources)):
            m = te & (sources == src)
            if m.any():
                tp2, fp2, fn2, pr2, rc2 = metrics(predict(X[m], head), y[m], best_thr)
                print(f"  {src:14} horns found {tp2:4d} of {int(y[m].sum()):4d}   false alarms {fp2:4d} of {int((1 - y[m]).sum()):5d}"
                      f"   precision {pr2:.2f}   recall {rc2:.2f}")
    if len(set(sources)) > 1:
        print("\nFalse alarms and horns found on the testing clips of each dataset, per threshold:")
        print("  threshold  " + "".join(f"{s:>26}" for s in sorted(set(sources))))
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            cells = []
            for src in sorted(set(sources)):
                m = te & (sources == src)
                tp2, fp2, fn2, pr2, rc2 = metrics(predict(X[m], head), y[m], thr)
                cells.append(f"{tp2}/{int(y[m].sum())} horns, {fp2}/{int((1 - y[m]).sum())} false")
            print(f"  {thr:9.2f}  " + "".join(f"{c:>26}" for c in cells))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, w=head[0], b=head[1], mean=head[2], std=head[3], threshold=best_thr,
             normalize=bool(args.normalize))
    print(f"Saved {args.out}. The detector uses it automatically when it starts.")


if __name__ == "__main__":
    main()
