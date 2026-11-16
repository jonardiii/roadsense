"""Try plate reading on ordinary photos (no camera or web system needed).

  python plate_test.py --images samples/photos --model models/plate.pt

Takes every .jpg/.png in the folder, finds the plate, reads it, and saves a copy with the result drawn
on it in the folder 'results/plate_test_out'. To get an accuracy figure, name each photo after its real plate,
for example  ABC1234.jpg,  ABC 1234.jpg  or  NAB5678_front.png  (a second photo of the same car: ABC1234_2.jpg).  The script then counts
how many it read exactly right.
"""
import argparse
import os
import re

import cv2

from detector.plates import formats
from detector.plates.ocr import ENGINES, make_ocr, sharpness
from detector.plates.reader import PlateDetector, read_best

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def expected_plate(name):
    """The plate a photo is named after: 'JAC 7183.jpg', 'JAC7183_front.jpg', 'JAC-7183 (2).jpg'."""
    stem = os.path.splitext(name)[0]
    part = re.sub(r"\(\d+\)", "", stem.split("_")[0])      # drop '_front' endings and '(2)' copies
    plate = re.sub(r"[\s-]", "", part).upper()
    return plate if formats.is_valid_format(plate) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--model", default="models/plate.pt")
    ap.add_argument("--out", default="results/plate_test_out")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=1280, help="picture size used to search for the plate (large phone photos need more than 640)")
    ap.add_argument("--candidates", type=int, default=3, help="how many plate-like boxes to try per photo (1 = old behaviour)")
    ap.add_argument("--cpu-ocr", action="store_true")
    ap.add_argument("--ocr", default="easyocr", choices=ENGINES, help="which text reader to use")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.images) if f.lower().endswith(EXT))
    if not files:
        raise SystemExit(f"No photos found in {a.images}")
    os.makedirs(a.out, exist_ok=True)
    detector = PlateDetector(a.model, a.device, imgsz=a.imgsz)
    ocr = make_ocr(a.ocr, gpu=not a.cpu_ocr)

    known = exact = found = 0
    print(f"{'photo':32} {'plate found':>11} {'sharp':>6}  {'OCR text':10} {'corrected':10} {'format':6} {'expected':9}")
    for name in files:
        image = cv2.imread(os.path.join(a.images, name))
        if image is None:
            print(f"{name:32} could not open")
            continue
        want = expected_plate(name)
        item = read_best(detector, ocr, image, a.candidates)
        hit = item is not None
        raw = fixed = ""
        sharp = 0.0
        label, colour = "no plate found", (0, 0, 255)
        if item:
            found += 1
            x1, y1, x2, y2 = item["box"]
            sharp = sharpness(item["crop"])
            raw, fixed = item["raw"], item["text"]
            label, colour = (fixed or "unreadable"), (0, 200, 0)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colour, 3)
            cv2.putText(image, label, (int(x1), max(25, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, colour, 3, cv2.LINE_AA)
        cv2.imwrite(os.path.join(a.out, name), image)
        mark = ""
        if want:
            known += 1
            ok = fixed == want
            exact += ok
            mark = "OK" if ok else "WRONG"
        print(f"{name:32} {'yes' if hit else 'NO':>11} {sharp:6.0f}  {raw:10} {fixed:10} "
              f"{'ok' if fixed and formats.is_valid_format(fixed) else '-':6} {want or '':9} {mark}")

    print(f"\nPlate found in {found} of {len(files)} photos.")
    if known:
        print(f"Read exactly right: {exact} of {known} photos with a known plate ({100.0 * exact / known:.0f}%).")
        print("This is one photo per vehicle. In real use the detector reads many frames of each vehicle and votes, "
              "so it should do better than this.")
    else:
        print("Name photos like ABC1234_01.jpg to get an accuracy figure.")
    print(f"Annotated copies are in '{a.out}'.")


if __name__ == "__main__":
    main()
