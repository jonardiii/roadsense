"""Get plate datasets into one YOLO training set.

  # 1. Kaggle "Car License Plate Detection" (images/ + annotations/*.xml in Pascal VOC format)
  python -m data_prep.prepare_dataset voc --images kaggle/images --annotations kaggle/annotations --out data/kaggle

  # 2. Datasets exported from Roboflow or CVAT in "YOLOv11" format already have the right layout.

  # 3. Combine them. Every class becomes one class, "plate". Repeat your own camera frames so they
  #    count for more than the foreign plates (here 3 times):
  python -m data_prep.prepare_dataset merge --inputs data/kaggle data/roboflow_ph data/own_frames --repeats 1 1 3 --out data/plates
"""
import argparse
import re
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS_TRAIN = ("train",)
SPLITS_VAL = ("val", "valid", "test")


def write_yaml(out, names=("plate",)):
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write(f"path: {os.path.abspath(out).replace(os.sep, '/')}\n")
        fh.write("train: images/train\nval: images/val\n")
        fh.write(f"nc: {len(names)}\nnames: {list(names)}\n")


def voc_box(xml_path):
    """Return (image width, height, [boxes]) from one Pascal VOC file."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(float(size.findtext("width"))), int(float(size.findtext("height")))
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((x1, y1, x2, y2))
    return w, h, boxes


def to_yolo_line(box, w, h):
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def make_dirs(out):
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            os.makedirs(os.path.join(out, kind, split), exist_ok=True)


def cmd_voc(args):
    make_dirs(args.out)
    names = sorted(f for f in os.listdir(args.images) if os.path.splitext(f)[1].lower() in IMAGE_EXT)
    random.Random(args.seed).shuffle(names)
    n_val = max(1, int(len(names) * args.val_fraction)) if len(names) > 1 else 0
    used = skipped = 0
    for i, name in enumerate(names):
        stem = os.path.splitext(name)[0]
        xml = os.path.join(args.annotations, stem + ".xml")
        if not os.path.exists(xml):
            skipped += 1
            continue
        w, h, boxes = voc_box(xml)
        lines = [ln for ln in (to_yolo_line(b, w, h) for b in boxes) if ln]
        if not lines:
            skipped += 1
            continue
        split = "val" if i < n_val else "train"
        shutil.copy2(os.path.join(args.images, name), os.path.join(args.out, "images", split, name))
        with open(os.path.join(args.out, "labels", split, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        used += 1
    write_yaml(args.out)
    print(f"Converted {used} images to {args.out} ({skipped} skipped: no annotation or no usable box).")


def split_dirs(root):
    """Yield (split, images dir, labels dir) for a YOLO-format dataset folder.

    Two layouts exist: root/images/<split> + root/labels/<split> (what this tool writes), and
    root/<split>/images + root/<split>/labels (what Roboflow exports).
    """
    for split in SPLITS_TRAIN + SPLITS_VAL:
        a = os.path.join(root, "images", split), os.path.join(root, "labels", split)
        b = os.path.join(root, split, "images"), os.path.join(root, split, "labels")
        for img, lab in (a, b):
            if os.path.isdir(img) and os.path.isdir(lab):
                yield split, img, lab
                break


def label_to_box(row):
    """One label row as (cx, cy, w, h). Handles box rows and polygon (segmentation) rows."""
    values = [float(v) for v in row[1:]]
    if len(values) == 4:
        return values
    if len(values) >= 6 and len(values) % 2 == 0:
        xs, ys = values[0::2], values[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    return None


def plate_class_ids(root):
    """Class numbers to keep from a dataset. A one-class set keeps everything. In a set with
    several classes (for example Motor, car, licenseplate) only the classes with "plate" in the
    name are kept, so cars and motorcycles are never turned into plates. Returns None for all."""
    path = os.path.join(root, "data.yaml")
    if not os.path.exists(path):
        return None
    text = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"names:\s*\[(.*?)\]", text, re.S)
    if m:
        names = [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
    else:
        names = [mm.group(1).strip().strip("'\"") for mm in re.finditer(r"^\s+\d+:\s*(.+)$", text, re.M)]
    if len(names) <= 1:
        return None
    keep = {i for i, n in enumerate(names) if "plate" in n.lower()}
    print(f"{root}: classes {names} -> keeping {[names[i] for i in sorted(keep)] or 'nothing'}")
    return keep


def cmd_merge(args):
    repeats = args.repeats or [1] * len(args.inputs)
    if len(repeats) != len(args.inputs):
        sys.exit("--repeats needs one number per input folder.")
    make_dirs(args.out)
    counts = {"train": 0, "val": 0}
    for tag, (root, rep) in enumerate(zip(args.inputs, repeats)):
        found = False
        keep = plate_class_ids(root)
        for split, img_dir, lab_dir in split_dirs(root):
            found = True
            target = "train" if split in SPLITS_TRAIN else "val"
            for name in sorted(os.listdir(img_dir)):
                stem, ext = os.path.splitext(name)
                if ext.lower() not in IMAGE_EXT:
                    continue
                lab = os.path.join(lab_dir, stem + ".txt")
                if not os.path.exists(lab):
                    continue
                with open(lab) as fh:
                    parsed = [label_to_box(r.split()) for r in fh.read().splitlines()
                              if r.strip() and (keep is None or int(float(r.split()[0])) in keep)]
                # every class becomes class 0, "plate"
                rows = [["0"] + [f"{v:.6f}" for v in box] for box in parsed if box]
                if not rows:
                    continue
                copies = rep if target == "train" else 1
                for c in range(copies):
                    new = f"d{tag}_{stem}" + (f"_r{c}" if c else "")
                    shutil.copy2(os.path.join(img_dir, name), os.path.join(args.out, "images", target, new + ext))
                    with open(os.path.join(args.out, "labels", target, new + ".txt"), "w") as fh:
                        fh.write("\n".join(" ".join(r) for r in rows) + "\n")
                    counts[target] += 1
        if not found:
            print(f"Warning: {root} has no train/val images folders. Skipped.")
    write_yaml(args.out)
    print(f"Merged into {args.out}: {counts['train']} training and {counts['val']} validation images "
          "(repeated copies included).")
    if counts["val"] == 0:
        print("Warning: no validation images. Add a dataset with a valid/test split, or use 'voc' (it splits).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("voc", help="convert Pascal VOC (XML) annotations to YOLO format")
    v.add_argument("--images", required=True)
    v.add_argument("--annotations", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--val-fraction", type=float, default=0.15)
    v.add_argument("--seed", type=int, default=0)
    v.set_defaults(func=cmd_voc)
    m = sub.add_parser("merge", help="combine YOLO-format datasets into one single-class set")
    m.add_argument("--inputs", nargs="+", required=True)
    m.add_argument("--repeats", nargs="+", type=int)
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_merge)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
