"""Combine Roboflow vehicle datasets into one set with the classes RoadSense reports.

   python prepare_vehicles.py --inputs data\\ph_vehicles1 data\\ph_vehicles2 --out data\\vehicles
   python prepare_vehicles.py --inputs data\\big_set --max-images 6000 --out data\\vehicles

Every dataset is a Roboflow export in YOLO format (it has a data.yaml and train/valid folders). The class
names in each one are matched to these nine: car, van, jeepney, tricycle, motorcycle, bus, truck, pickup,
multicab. Add "*3" after a folder (data\\own_frames*3) to use its training images three times: do that for a
small set of your own labelled frames, so the model pays attention to them.
Classes that match none of them (bicycle, e-bike, person...) are dropped, and their boxes with them.
The mapping used for each dataset is printed, so check it before you train.
"""
import argparse
import os
import random
import re
import shutil
import sys

from prepare_dataset import IMAGE_EXT, SPLITS_TRAIN, label_to_box, split_dirs

CLASSES = ["car", "van", "jeepney", "tricycle", "motorcycle", "bus", "truck", "pickup", "multicab"]

# first match wins, so the specific words come first
RULES = [
    ("multicab", "multicab"), ("multi-cab", "multicab"), ("multi cab", "multicab"),
    ("jeepney", "jeepney"), ("jeep", "jeepney"), ("jitney", "jeepney"), ("puj", "jeepney"),
    ("tricycle", "tricycle"), ("tuk", "tricycle"), ("trike", "tricycle"), ("pedicab", None),
    ("e-bike", None), ("ebike", None), ("bicycle", None), ("cyclist", None),
    ("motor", "motorcycle"), ("scooter", "motorcycle"),
    ("uv express", "van"), ("uv-express", "van"), ("uvexpress", "van"),
    ("minibus", "van"), ("bus", "bus"),
    ("van", "van"), ("l300", "van"), ("urvan", "van"),
    ("pickup", "pickup"), ("pick-up", "pickup"), ("pick up", "pickup"), ("hilux", "pickup"),
    ("strada", "pickup"), ("navara", "pickup"),
    ("trailer", "truck"), ("flatbed", "truck"), ("truck", "truck"), ("lorry", "truck"),
    ("suv", "car"), ("sedan", "car"), ("hatchback", "car"),
    ("taxi", "car"), ("private", "car"), ("mpv", "car"), ("car", "car"),
]


def read_names(root):
    path = os.path.join(root, "data.yaml")
    if not os.path.exists(path):
        sys.exit(f"{root} has no data.yaml. Download the dataset from Roboflow in YOLO format.")
    text = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"names:\s*\[(.*?)\]", text, re.S)
    if m:
        return [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
    return [mm.group(1).strip().strip("'\"") for mm in re.finditer(r"^\s+\d+:\s*(.+)$", text, re.M)]


def map_class(name):
    low = name.lower().strip()
    if low in CLASSES:
        return low
    for key, target in RULES:
        if key in low:
            return target
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-images", type=int, default=0,
                    help="most training images to take from each dataset (0 = all). Use it for very big sets.")
    ap.add_argument("--val-fraction", type=float, default=0.1,
                    help="a dataset with only a train folder has this share of its images set aside for testing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    counts = {"train": 0, "val": 0}
    boxes = {c: 0 for c in CLASSES}
    for tag, spec in enumerate(args.inputs):
        root, _, times = spec.partition("*")
        repeat = max(1, int(times)) if times.isdigit() else 1
        names = read_names(root)
        mapping = {i: map_class(n) for i, n in enumerate(names)}
        print(f"\n{root}")
        for i, n in enumerate(names):
            print(f"   {n!r:28} -> {mapping[i] or 'dropped'}")
        found = list(split_dirs(root))
        has_val = any(sp not in SPLITS_TRAIN for sp, _, _ in found)
        if found and not has_val:
            print(f"   no valid/test folder: {args.val_fraction:.0%} of the images are set aside for testing")
        for split, img_dir, lab_dir in found:
            target = "train" if split in SPLITS_TRAIN else "val"
            files = sorted(f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXT)
            if args.max_images and target == "train" and len(files) > args.max_images:
                files = sorted(rng.sample(files, args.max_images))
            held_out = set()
            if not has_val and target == "train":
                held_out = set(rng.sample(files, int(len(files) * args.val_fraction)))
            for name in files:
                target = "val" if name in held_out else ("train" if split in SPLITS_TRAIN else "val")
                stem, ext = os.path.splitext(name)
                lab = os.path.join(lab_dir, stem + ".txt")
                if not os.path.exists(lab):
                    continue
                rows = []
                for r in open(lab).read().splitlines():
                    parts = r.split()
                    if not parts:
                        continue
                    cls = mapping.get(int(float(parts[0])))
                    box = label_to_box(parts)
                    if cls and box:
                        rows.append([str(CLASSES.index(cls))] + [f"{v:.6f}" for v in box])
                        boxes[cls] += 1 if target == "train" else 0
                if not rows:
                    continue
                for k in range(repeat if target == "train" else 1):
                    new = f"v{tag}_{stem}" + (f"_r{k}" if k else "")
                    shutil.copy2(os.path.join(img_dir, name), os.path.join(args.out, "images", target, new + ext))
                    with open(os.path.join(args.out, "labels", target, new + ".txt"), "w") as fh:
                        fh.write("\n".join(" ".join(r) for r in rows) + "\n")
                    counts[target] += 1
                    if k:
                        for r in rows:
                            boxes[CLASSES[int(r[0])]] += 1
    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write(f"path: {os.path.abspath(args.out).replace(os.sep, '/')}\n")
        fh.write("train: images/train\nval: images/val\n")
        fh.write(f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")
    print(f"\nMerged into {args.out}: {counts['train']} training and {counts['val']} validation images.")
    print("Training boxes per class:", ", ".join(f"{c} {n}" for c, n in boxes.items()))
    thin = [c for c, n in boxes.items() if n < 300]
    if thin:
        print("Warning: few examples of", ", ".join(thin), "- the model will be weak on those. Add a dataset with more.")
    if counts["val"] == 0:
        print("Warning: no validation images. Use a dataset that has a valid/test folder.")


if __name__ == "__main__":
    main()
