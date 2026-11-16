"""Plate reader used by the pipeline: for each tracked vehicle, find the plate on some frames,
read it, keep every reading, and vote when the vehicle has been measured."""
import logging

from . import formats
from .ocr import sharpness
from .voting import PlateVote

log = logging.getLogger("roadsense.plates")


class PlateDetector:
    """YOLO model trained to find licence plates (see training/train_plate_model.py)."""

    def __init__(self, weights, device="0", imgsz=640, conf=0.3):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install the detector packages first: pip install -r requirements.txt") from exc
        self.model = YOLO(weights)
        self.kw = dict(conf=conf, imgsz=imgsz, device=device, verbose=False)

    def detect(self, image):
        """The most confident plate in the image as ((x1, y1, x2, y2), confidence), or None."""
        found = self.candidates(image, max_n=1)
        return found[0] if found else None

    def candidates(self, image, max_n=3):
        """Up to max_n plate-like boxes, most confident first, as [((x1, y1, x2, y2), confidence), ...].

        Boxes whose shape cannot be a plate (a round sticker, a long thin exhaust pipe) are dropped."""
        result = self.model.predict(image, **self.kw)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        order = result.boxes.conf.argsort(descending=True).tolist()
        out = []
        for i in order:
            box = tuple(result.boxes.xyxy[i].tolist())
            if plausible_shape(box):
                out.append((box, float(result.boxes.conf[i])))
            if len(out) >= max_n:
                break
        return out


def plausible_shape(box, lowest=1.1, highest=5.0):
    """True if width / height looks like a licence plate (cars about 2:1, motorcycles about 1.3:1)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return h > 0 and w > 0 and lowest <= w / h <= highest


def crop_plate(image, box):
    x1, y1, x2, y2 = box
    pad = int(0.06 * (x2 - x1))
    h, w = image.shape[:2]
    return image[max(0, int(y1) - pad):min(h, int(y2) + pad), max(0, int(x1) - pad):min(w, int(x2) + pad)]


def find_candidates(detector, image, max_n=3):
    """Candidate plate boxes from any detector (older ones only have detect())."""
    if hasattr(detector, "candidates"):
        return detector.candidates(image, max_n)
    hit = detector.detect(image)
    return [hit] if hit else []


def read_best(detector, ocr, image, max_n=3, keep=None):
    """Try the plate candidates in order of confidence; stop at the first whose text is a valid plate.

    A muffler or sticker can score higher than the real plate, but its text is rarely a valid plate,
    so the next candidate gets its turn. If none reads as a valid plate, the most confident
    candidate's reading is returned. keep(crop, box) may reject a candidate before it is read.

    Returns None, or a dict with box, plate_conf, crop, raw, text (format-corrected), ocr_conf, valid."""
    fallback = None
    for box, plate_conf in find_candidates(detector, image, max_n):
        crop = crop_plate(image, box)
        if crop.size == 0 or (keep is not None and not keep(crop, box)):
            continue
        out = ocr.read(crop)
        raw, ocr_conf = out if out else ("", 0.0)
        text = formats.correct(raw)[0] if raw else ""
        item = dict(box=box, plate_conf=plate_conf, crop=crop, raw=raw, text=text,
                    ocr_conf=ocr_conf, valid=bool(text) and formats.is_valid_format(text))
        if item["valid"]:
            return item
        if fallback is None:
            fallback = item
    return fallback


class PlateReader:
    def __init__(self, detector, ocr, verified=(), min_confidence=0.6, every_n_frames=2,
                 max_reads=15, max_attempts=45, min_vehicle_width=120, min_plate_width=50,
                 min_sharpness=25.0, max_candidates=3, require_valid_format=False):
        self.max_candidates = max_candidates
        self.require_valid_format = require_valid_format
        self.detector = detector
        self.ocr = ocr
        self.min_confidence = min_confidence
        self.every_n = max(1, every_n_frames)
        self.max_reads = max_reads
        self.max_attempts = max_attempts            # caps the work spent on one vehicle
        self.min_vehicle_width = min_vehicle_width  # far-away vehicles are too small to read
        self.min_plate_width = min_plate_width
        self.min_sharpness = min_sharpness
        self._verified = set(verified)
        self._votes, self._frames, self._attempts = {}, {}, {}
        self._boxes = {}   # track id -> where the plate was last seen, for the live picture

    def set_verified(self, plates):
        self._verified = set(plates or ())

    def observe(self, track_id, image, box):
        n = self._frames.get(track_id, 0)
        self._frames[track_id] = n + 1
        if n % self.every_n or self._attempts.get(track_id, 0) >= self.max_attempts:
            return
        x1, y1, x2, y2 = (int(v) for v in box)
        width, height = x2 - x1, y2 - y1
        if width < self.min_vehicle_width:
            return
        self._attempts[track_id] = self._attempts.get(track_id, 0) + 1

        # Plates sit in the lower part of a vehicle, so search there.
        ih, iw = image.shape[:2]
        top = max(0, y1 + int(0.35 * height))
        region = image[top:min(ih, y2), max(0, x1):min(iw, x2)]
        if region.size == 0:
            return
        def usable(crop, box):
            return box[2] - box[0] >= self.min_plate_width and sharpness(crop) >= self.min_sharpness

        item = read_best(self.detector, self.ocr, region, self.max_candidates, keep=usable)
        if not item or not item["raw"]:
            return
        pb = item["box"]
        self._boxes[track_id] = (max(0, x1) + int(pb[0]), top + int(pb[1]), max(0, x1) + int(pb[2]), top + int(pb[3]))
        text, conf, plate_w = item["raw"], item["ocr_conf"], item["box"][2] - item["box"][0]
        sharp, plate_conf = sharpness(item["crop"]), item["plate_conf"]
        # How much to trust this frame: a larger, sharper, surely-a-plate view counts for more.
        quality = min(1.0, plate_w / 140.0) * min(1.0, 0.4 + sharp / 150.0) * plate_conf
        self._votes.setdefault(track_id, PlateVote(self.max_reads)).add(text, conf, quality)

    def guess(self, track_id):
        """Best plate so far and its confidence, even if too low to use."""
        vote = self._votes.get(track_id)
        c = vote.consensus(self._verified) if vote else None
        return (c.text, c.confidence) if c else (None, None)

    def accepted(self, track_id):
        """False when the reading is not in any Philippine plate format and no registered plate backs
        it up. A sticker or a sign on a truck can read as clear text, but it is not a plate."""
        if not self.require_valid_format:
            return True
        vote = self._votes.get(track_id)
        c = vote.consensus(self._verified) if vote else None
        return bool(c) and (c.verified or formats.is_valid_format(c.text))

    def visible(self, track_id):
        """(box, text, confidence) to draw on the live picture, only for readings that would be sent."""
        box = self._boxes.get(track_id)
        text, conf = self.guess(track_id)
        if not box or not text or not self.accepted(track_id) or conf < self.min_confidence:
            return None   # a doubtful reading is not drawn: it would only mislead
        return box, text, conf

    def result(self, track_id):
        """(plate, confidence) if confident enough to send, otherwise (None, confidence)."""
        text, conf = self.guess(track_id)
        if text is None:
            return None, None
        if not self.accepted(track_id):
            log.info("Track %s: %s is not a Philippine plate format. Sending without a plate.", track_id, text)
            return None, conf
        if conf >= self.min_confidence:
            return text, conf
        log.info("Track %s: plate unclear (best guess %s, %.2f). Sending without a plate.",
                 track_id, text, conf)
        return None, conf

    def last_box(self, track_id):
        """(x1, y1, x2, y2) of the plate most recently found on this vehicle, in picture pixels."""
        return self._boxes.get(track_id)

    def forget(self, track_id):
        for d in (self._votes, self._frames, self._attempts, self._boxes):
            d.pop(track_id, None)
