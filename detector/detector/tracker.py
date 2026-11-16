"""YOLO11 vehicle detection with tracking. Each vehicle keeps one ID across frames."""
from dataclasses import dataclass


@dataclass
class Detection:
    track_id: int
    box: tuple          # x1, y1, x2, y2 in pixels
    conf: float
    cls: int
    name: str


class YoloTracker:
    def __init__(self, weights="yolo11s.pt", device="0", imgsz=960, conf=0.35, iou=0.5,
                 classes=(2, 3, 5, 7), tracker="bytetrack.yaml"):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install the detector packages first: pip install -r requirements.txt") from exc
        self.model = YOLO(weights)
        # an empty list means every class of the model (a model trained on RoadSense vehicle classes)
        self.kw = dict(conf=conf, iou=iou, imgsz=imgsz, classes=list(classes) or None, device=device,
                       tracker=tracker, persist=True, verbose=False)

    def track(self, frame):
        result = self.model.track(frame, **self.kw)[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()
        names = result.names
        return [Detection(i, tuple(b), c, k, names.get(k, str(k)))
                for i, b, c, k in zip(ids, xyxy, confs, classes)]
