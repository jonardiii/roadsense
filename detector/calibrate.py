"""Set up the speed zone.  Run it once per camera position.

1. Mark a rectangle on the road with chalk or cones and measure it with a tape.
   3 m wide and 6 to 8 m long works well. The long side must run along the road.
2. Grab a frame from the camera (or use a video of the road with the marks visible).
3. Click the four corners in this order, as the camera sees them:
       near-left, near-right, far-right, far-left
4. Press S to save, R to click again, Q to quit.

   python calibrate.py --source clip.mp4 --width 3 --length 6
   python calibrate.py --image frame.png --width 3 --length 6 --points "310,650 990,650 800,300 480,300"
"""
import argparse
import os
import sys

import cv2

from detector.speed import Calibration

ORDER = ["near-left", "near-right", "far-right", "far-left"]


def grab_frame(source, frame_no):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"Could not open {source}")
    image = None
    for i in range(max(frame_no, 1)):
        ok, img = cap.read()
        if not ok:
            break
        image = img
    cap.release()
    if image is None:
        sys.exit("Could not read a frame from the source.")
    return image


def parse_points(text):
    pts = []
    for chunk in text.replace(";", " ").split():
        x, y = chunk.split(",")
        pts.append((float(x), float(y)))
    if len(pts) != 4:
        sys.exit("--points needs exactly four x,y pairs.")
    return pts


def click_points(image):
    pts = []
    win = "Click: " + ", ".join(ORDER) + "   (R redo, S save, Q quit)"

    def draw():
        view = image.copy()
        for i, p in enumerate(pts):
            cv2.circle(view, p, 7, (0, 0, 255), -1)
            cv2.putText(view, ORDER[i], (p[0] + 10, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
        if len(pts) == 4:
            cv2.polylines(view, [__import__("numpy").array(pts)], True, (0, 220, 255), 2)
        cv2.imshow(win, view)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            draw()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    draw()
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            sys.exit("Cancelled. Nothing was saved.")
        if key == ord("r"):
            pts.clear()
            draw()
        if key == ord("s") and len(pts) == 4:
            cv2.destroyAllWindows()
            return [tuple(map(float, p)) for p in pts]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="video file, RTSP address or camera number")
    src.add_argument("--image", help="a still image instead of a video")
    ap.add_argument("--frame", type=int, default=30, help="which frame to use from a video (default 30)")
    ap.add_argument("--width", type=float, required=True, help="rectangle width across the road, metres")
    ap.add_argument("--length", type=float, required=True, help="rectangle length along the road, metres")
    ap.add_argument("--points", help='corners without a window: "x,y x,y x,y x,y"')
    ap.add_argument("--out", default="calibration.json")
    args = ap.parse_args()

    image = cv2.imread(args.image) if args.image else grab_frame(args.source, args.frame)
    if image is None:
        sys.exit("Could not read the image.")
    h, w = image.shape[:2]
    pts = parse_points(args.points) if args.points else click_points(image)

    try:
        cal = Calibration.from_rectangle(pts, args.width, args.length, (w, h))
    except ValueError as exc:
        sys.exit(str(exc))
    cal.save(args.out)
    preview = os.path.splitext(args.out)[0] + "_preview.png"
    cv2.imwrite(preview, cal.draw(image))
    print(f"Saved {args.out} ({w}x{h} frame, {args.width:g} x {args.length:g} m zone).")
    print(f"Open {preview} and check that the yellow zone sits on your marked rectangle and the "
          "white lines are one metre apart on the road.")
    print("If you change the camera's position or angle, run this again.")


if __name__ == "__main__":
    main()
