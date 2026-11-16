"""Pull sharp frames out of a video to label as plate training images.

   python -m data_prep.extract_frames --video clip.mp4 --out data/own_raw --every 5

Upload the frames to Roboflow or CVAT, draw a box around each plate, and export as YOLOv11.
Frames with a vehicle close enough to read the plate are the useful ones.
"""
import argparse
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=5, help="look at every Nth frame")
    ap.add_argument("--min-sharpness", type=float, default=60.0, help="skip blurrier frames")
    ap.add_argument("--max", type=int, default=400, help="stop after this many frames")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    os.makedirs(args.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video))[0]
    idx = saved = blurry = 0
    while saved < args.max:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every == 0:
            sharp = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if sharp >= args.min_sharpness:
                cv2.imwrite(os.path.join(args.out, f"{base}_{idx:06d}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
            else:
                blurry += 1
        idx += 1
    cap.release()
    print(f"Saved {saved} frames to {args.out} ({blurry} skipped as blurry, {idx} frames read).")


if __name__ == "__main__":
    main()
