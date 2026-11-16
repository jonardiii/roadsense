"""Detector settings, read from a JSON file (see detector_config.example.json)."""
import json
import os
from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class Config:
    # Web system
    api_url: str = "http://127.0.0.1:5000"
    api_key: str = ""                      # or set the ROADSENSE_API_KEY environment variable

    # Video: an RTSP address, a video file path, or a camera number such as 0
    source: str = ""
    calibration: str = "calibration.json"

    # Vehicle detection
    model: str = "yolo11s.pt"
    device: str = "0"                      # "0" = first GPU, "cpu" = no GPU
    imgsz: int = 960
    conf: float = 0.35
    iou: float = 0.5
    tracker: str = "bytetrack.yaml"
    vehicle_classes: list = field(default_factory=lambda: [2, 3, 5, 7])  # car, motorcycle, bus, truck

    # Speed. A vehicle is flagged above (speed limit + margin). Choose the margin from the
    # measured error of your own validation runs, not by guessing.
    speed_margin_kmh: float = 0.0
    min_samples: int = 6
    min_duration_s: float = 0.4
    min_distance_m: float = 2.0
    max_residual_m: float = 0.5

    # Plate reading. Leave plate_model empty to run without it (violations are then sent without
    # a plate and the administrator reads it from the photo).
    plate_model: str = ""                  # trained plate detector, e.g. "models/plate.pt"
    plate_conf: float = 0.3
    ocr_engine: str = "easyocr"            # "easyocr" or "fastplate" (compare them with distance_test.py)
    ocr_gpu: bool = True
    require_valid_format: bool = True      # a reading that is not a Philippine plate format (a sticker, a sign) is not used
    min_plate_confidence: float = 0.6      # below this the plate is not sent, only shown on the photo
    plate_candidates: int = 3              # plate-like boxes tried per frame (a muffler or sticker may score higher)
    plate_every_n_frames: int = 2          # read the plate on every Nth frame of each vehicle
    max_plate_reads: int = 15
    min_vehicle_width_px: int = 120        # smaller vehicles are too far away to read
    min_plate_width_px: int = 50
    min_plate_sharpness: float = 25.0

    # Horn detection from the microphone (off until horn_enabled is true)
    horn_enabled: bool = False
    audio_device: str = ""                 # empty = the default microphone; or a number or name (see horn_tool.py devices)
    audio_rate: int = 16000
    horn_min_excess_db: float = 10.0       # how much louder than the background a horn must be
    horn_min_tonal: float = 0.6
    horn_min_similarity: float = 0.85
    horn_min_duration_s: float = 0.2
    horn_cooldown_s: float = 2.0           # beeps this close together count as one horn
    video_lag_s: float = 1.0               # how far the camera picture runs behind the sound (see README)
    horn_model: str = "models/yamnet.onnx"    # pre-trained sound model that tells a horn from shouting (optional; see README)
    horn_head: str = "models/horn_head.npz"   # layer fine-tuned on horn recordings (made by horn_train_head.py)
    horn_head_min_score: float = 0.0       # threshold for the fine-tuned layer; 0 = use the one saved with it
    horn_model_min_score: float = 0.3      # how sure the sound model must be that it is a horn (0 to 1)
    horn_zone_only: bool = True            # a horn counts only for a vehicle inside the calibrated area
    horn_min_box_px: int = 80              # vehicles narrower than this in the picture are too far to be the source
    mic_offset_db: Optional[float] = None  # set to also send an estimated dB level (see README)

    # Housekeeping
    heartbeat_s: float = 30.0
    config_refresh_s: float = 60.0
    outbox_dir: str = "outbox"
    jpeg_quality: int = 90
    show_preview: bool = False
    parked_after_s: float = 15.0           # a vehicle in view this long without moving is parked and ignored (0 = off)
    camera_audio_off: bool = True          # never take the sound of the RTSP camera: only the external microphone is used
    clip_enabled: bool = True              # save a short video with each violation (needs: pip install imageio-ffmpeg)
    clip_before_s: float = 7.0             # seconds of video before the violation
    clip_after_s: float = 3.0              # seconds after it
    clip_width: int = 1280
    clip_fps: float = 12.0
    live_port: int = 5001                  # live video for the web page, 0 turns it off
    live_host: str = "127.0.0.1"           # this laptop only; "0.0.0.0" lets other devices on the network watch


def load(path):
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown setting(s) in {path}: {', '.join(unknown)}")
    cfg = Config(**raw)
    cfg.api_key = os.environ.get("ROADSENSE_API_KEY", cfg.api_key)
    return cfg
