"""How small can a plate get before it can't be read? (no camera needed)

  python distance_test.py --images samples/photos --model models/plate.pt

Takes your clear photos, finds the plate once at full size, then shrinks each photo so the plate becomes
160, 120, 100, 80, 60, 50, 40 and 30 pixels wide (as if the vehicle were farther from the camera), and
tries to find and read the plate again. Name each photo after its real plate, like NAB1234_01.jpg, so
the result can be counted as right or wrong.

Shrinking a photo is kinder than real distance (no lens blur, no camera noise). Add --camera-like to
blur and compress the small copies the way a cheap camera does, which is closer to real life. Treat
the numbers as a best case, and repeat the test with real footage from your camera on site.
"""
import argparse
import os

import cv2
import numpy as np

from detector.plates import formats
from detector.plates.ocr import ENGINES, make_ocr
from detector.plates.reader import PlateDetector, read_best
from plate_test import EXT, expected_plate


def camera_like(image, seed):
    """Soften and compress a small picture the way a budget camera stream does."""
    rng = np.random.default_rng(seed)
    out = cv2.GaussianBlur(image, (0, 0), 0.8)
    noise = rng.normal(0, 3.0, out.shape)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 45])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else out


def motion_blur(image, length):
    """Smear the picture sideways over `length` pixels, like a vehicle moving while the shutter is open."""
    if length < 2:
        return image
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0 / length
    return cv2.filter2D(image, -1, kernel)


def read_plate(detector, ocr, image, max_n=3):
    item = read_best(detector, ocr, image, max_n)
    return (item["box"], item["plate_conf"]) if item else None, (item["text"] if item else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--model", default="models/plate.pt")
    ap.add_argument("--widths", type=int, nargs="+", default=[160, 120, 100, 80, 60, 50, 40, 30])
    ap.add_argument("--camera-like", action="store_true", help="blur, noise and compression on the small copies")
    ap.add_argument("--motion-blur", type=float, default=0.0, metavar="FRACTION",
                    help="smear each small copy sideways by this fraction of the plate width, for example 0.1 "
                         "(a moving vehicle). Try 0.05, 0.1 and 0.2.")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=1280, help="picture size used to search for the plate")
    ap.add_argument("--candidates", type=int, default=3, help="how many plate-like boxes to try per photo (1 = old behaviour)")
    ap.add_argument("--cpu-ocr", action="store_true")
    ap.add_argument("--ocr", default="easyocr", choices=ENGINES, help="which text reader to use")
    a = ap.parse_args()

    detector = PlateDetector(a.model, a.device, imgsz=a.imgsz)
    ocr = make_ocr(a.ocr, gpu=not a.cpu_ocr)
    photos = []
    for name in sorted(os.listdir(a.images)):
        want = expected_plate(name)
        if not name.lower().endswith(EXT) or not want:
            continue
        image = cv2.imread(os.path.join(a.images, name))
        hit = read_plate(detector, ocr, image, a.candidates)[0] if image is not None else None
        if hit:
            photos.append((name, want, image, hit[0][2] - hit[0][0]))
    if not photos:
        raise SystemExit("No usable photos. Name them after the real plate, like NAB1234_01.jpg, and make "
                         "sure the plate is large and clear in each.")
    print(f"Text reader: {a.ocr}")
    print(f"{len(photos)} photos with a known plate and a plate found at full size.\n")
    print(f"{'plate width':>11}  {'plate found':>11}  {'read exactly right':>18}")
    rows = []
    for width in a.widths:
        found = right = total = 0
        for k, (name, want, image, w0) in enumerate(photos):
            if w0 < width:
                continue            # the photo is already smaller than this size
            total += 1
            scale = width / w0
            small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if a.motion_blur:
                small = motion_blur(small, max(2, round(width * a.motion_blur)))
            if a.camera_like:
                small = camera_like(small, k)
            hit, text = read_plate(detector, ocr, small, a.candidates)
            found += bool(hit)
            right += text == want
        rows.append((width, total, found, right))
        if total:
            print(f"{width:>8} px  {100 * found // total:>10}%  {100 * right // total:>17}%   ({total} photos)")
        else:
            print(f"{width:>8} px  no photo is big enough for this size")
    good = [w for w, t, f, r in rows if t and r / t >= 0.8]
    print()
    if good:
        print(f"Smallest plate width read exactly right on at least 80% of photos: {min(good)} px.")
    else:
        print("No size reached 80% exactly right. Use clearer photos or a larger plate in the photo.")
    print("Plan the camera so plates are at least that wide as the vehicle passes through the zone.")


if __name__ == "__main__":
    main()
