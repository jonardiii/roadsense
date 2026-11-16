"""Reading the characters from a cropped plate image."""
import cv2
import numpy as np

ALLOWED = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def sharpness(plate_bgr):
    """Blur measure (variance of the Laplacian) on a fixed-size grey copy. Higher is sharper."""
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY) if plate_bgr.ndim == 3 else plate_bgr
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (160, max(8, int(160 * h / max(w, 1)))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def preprocess(plate_bgr, target_height=96):
    """Enlarge, even out the lighting, and pad the edges so end characters aren't clipped."""
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY) if plate_bgr.ndim == 3 else plate_bgr
    h, w = gray.shape[:2]
    scale = min(6.0, max(1.0, target_height / max(h, 1)))
    gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    return cv2.copyMakeBorder(gray, 8, 8, 12, 12, cv2.BORDER_REPLICATE)


def join_results(results):
    """Turn OCR boxes into one string, reading two-line plates top line first.

    results: [(corner points, text, confidence), ...]. Returns (text, confidence) or None.
    """
    items = []
    for points, text, conf in results:
        pts = np.asarray(points, dtype=float)
        items.append((pts[:, 1].mean(), pts[:, 0].mean(), pts[:, 1].max() - pts[:, 1].min(), text, conf))
    if not items:
        return None
    # The plate number is the biggest text on the plate. The small print (the word PHILIPPINES,
    # a year sticker, a dealer frame) is not part of it, so drop anything much shorter.
    tallest = max(it[2] for it in items)
    items = [it for it in items if it[2] >= 0.55 * tallest]
    items.sort(key=lambda it: it[0])
    median_h = float(np.median([it[2] for it in items])) or 1.0
    lines, current = [], [items[0]]
    for prev, it in zip(items, items[1:]):
        if it[0] - prev[0] > 0.6 * median_h:
            lines.append(current)
            current = []
        current.append(it)
    lines.append(current)
    ordered = [it for line in lines for it in sorted(line, key=lambda it: it[1])]
    text = "".join(it[3] for it in ordered)
    chars = sum(len(it[3]) for it in ordered)
    conf = sum(it[4] * len(it[3]) for it in ordered) / max(chars, 1)
    return (text, float(conf)) if text else None


class EasyOcr:
    """EasyOCR restricted to capital letters and digits."""

    def __init__(self, gpu=True):
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("Install the OCR package first: pip install easyocr") from exc
        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def read(self, plate_bgr):
        image = preprocess(plate_bgr)
        results = self.reader.readtext(image, allowlist=ALLOWED, detail=1, paragraph=False,
                                       decoder="greedy", batch_size=1)
        return join_results(results)


class FastPlateOcr:
    """A reader trained only on licence plate crops (the fast-plate-ocr package, MIT licence).

    Same interface as EasyOcr. It runs on the CPU in a few milliseconds, so no GPU is needed.
    Install with:  pip install "fast-plate-ocr[onnx]"
    """

    def __init__(self, model="cct-s-v2-global-model", recognizer=None):
        if recognizer is None:
            try:
                from fast_plate_ocr import LicensePlateRecognizer
            except ImportError as exc:
                raise RuntimeError('Install the plate reader first: pip install "fast-plate-ocr[onnx]"') from exc
            recognizer = LicensePlateRecognizer(model, device="cpu")
        self.recognizer = recognizer

    def read(self, plate_bgr):
        rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB) if plate_bgr.ndim == 3 else plate_bgr
        pred = self.recognizer.run(rgb, return_confidence=True)[0]
        text = "".join(c for c in str(pred.plate).upper() if c in ALLOWED)
        if not text:
            return None
        probs = getattr(pred, "char_probs", None)
        conf = float(np.mean(np.asarray(probs, dtype=float)[:len(text)])) if probs is not None and len(probs) else 0.5
        return text, conf


ENGINES = ("easyocr", "fastplate")


def make_ocr(engine="easyocr", gpu=True):
    """The text reader chosen in the config (ocr_engine)."""
    if engine == "easyocr":
        return EasyOcr(gpu=gpu)
    if engine == "fastplate":
        return FastPlateOcr()
    raise ValueError(f"Unknown ocr_engine '{engine}'. Choose one of: {', '.join(ENGINES)}")
