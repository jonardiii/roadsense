"""Frames with timestamps, from an RTSP camera, a webcam or a video file."""
import collections
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np

log = logging.getLogger("roadsense.video")
PH_TZ = timezone(timedelta(hours=8))


@dataclass
class Frame:
    index: int
    t: float            # seconds on a steady clock; use differences only
    wall: datetime      # real time, for the violation record
    image: np.ndarray


def parse_source(source):
    source = str(source).strip()
    return int(source) if source.isdigit() else source


class FrameSource:
    def __init__(self, source, realtime=False, reconnect_delay=3.0, camera_audio_off=True):
        self.source = parse_source(source)
        self.camera_audio_off = camera_audio_off
        self.realtime = realtime            # files: play at normal speed instead of as fast as possible
        self.reconnect_delay = reconnect_delay
        self._stop = threading.Event()
        self.is_file = isinstance(self.source, str) and os.path.isfile(self.source)

    def close(self):
        self._stop.set()

    def frames(self):
        if self.is_file:
            yield from self._file_frames()
        else:
            yield from self._live_frames()

    # ---- video file ----
    def _file_frames(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open the video file: {self.source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        start = datetime.now(PH_TZ)
        began = time.monotonic()
        idx = 0
        try:
            while not self._stop.is_set():
                ok, image = cap.read()
                if not ok:
                    break
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                t = ms / 1000.0 if ms and ms > 0 else idx / fps
                if self.realtime:
                    delay = t - (time.monotonic() - began)
                    if delay > 0:
                        time.sleep(delay)
                yield Frame(idx, t, start + timedelta(seconds=t), image)
                idx += 1
        finally:
            cap.release()

    # ---- live stream ----
    def _no_camera_audio(self):
        """Ask for the camera's picture only. The sound of the system comes from the external microphone;
        the camera's own microphone is never requested, recorded or heard (OpenCV also never plays sound)."""
        if not (self.camera_audio_off and isinstance(self.source, str)
                and self.source.lower().startswith(("rtsp://", "rtsps://"))):
            return
        options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
        if "allowed_media_types" not in options:       # keep what the user set, or OpenCV's own default (tcp)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (options or "rtsp_transport;tcp") + "|allowed_media_types;video"

    def _open(self):
        self._no_camera_audio()
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reader(self, buf):
        """Read continuously so the camera's buffer never backs up. Each frame is stamped the
        moment it arrives, so a slow consumer drops old frames instead of skewing the times."""
        cap = self._open()
        idx = 0
        while not self._stop.is_set():
            if not cap.isOpened():
                log.warning("Camera not available. Retrying in %.0f s.", self.reconnect_delay)
                time.sleep(self.reconnect_delay)
                cap.release()
                cap = self._open()
                continue
            ok, image = cap.read()
            if not ok:
                log.warning("Lost the camera stream. Reconnecting.")
                cap.release()
                time.sleep(self.reconnect_delay)
                cap = self._open()
                continue
            buf.append(Frame(idx, time.monotonic(), datetime.now(PH_TZ), image))
            idx += 1
        cap.release()

    def _live_frames(self):
        buf = collections.deque(maxlen=60)
        thread = threading.Thread(target=self._reader, args=(buf,), daemon=True)
        thread.start()
        while not self._stop.is_set():
            try:
                yield buf.popleft()
            except IndexError:
                time.sleep(0.003)
