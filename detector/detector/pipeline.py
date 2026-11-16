"""Frame in, violations out. Ties tracking, speed, evidence photos and the web system together."""
import collections
import logging
from datetime import datetime

import cv2

log = logging.getLogger("roadsense.pipeline")


# Only these kinds of thing are ever measured or reported. People, animals, bicycles and anything
# else that moves are ignored, whatever the detector model is able to see.
VEHICLE_KINDS = ("car", "van", "jeepney", "tricycle", "motorcycle", "bus", "truck", "pickup", "multicab")


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class NullPlateReader:
    """Placeholder until plate reading is added: every vehicle has no plate."""

    def observe(self, track_id, image, box):
        pass

    def result(self, track_id):
        return None, None

    def guess(self, track_id):
        return None, None

    def last_box(self, track_id):
        return None

    def visible(self, track_id):
        return None

    def forget(self, track_id):
        pass


class Pipeline:
    def __init__(self, tracker, estimator, sender, speed_limit_kmh=10.0, margin_kmh=0.0,
                 plate_reader=None, jpeg_quality=90, edge_px=3, video_lag_s=1.0, horn_min_box_px=80,
                 mic_offset_db=None, clips=None, parked_after_s=15.0,
                 horn_min_level_db=None, horn_min_seconds=0.0, horn_zone=None):
        self.tracker = tracker
        self.estimator = estimator
        self.sender = sender
        self.clips = clips      # ClipRecorder or None: a short video is sent with each violation
        self._stamp = 0.0
        self.parked_after_s = parked_after_s   # 0 = never treat a vehicle as parked
        self._stills = []   # places where a vehicle has stayed: {box, first seen, last seen}
        self._stills_pruned = 0.0
        self.speed_limit_kmh = speed_limit_kmh
        self.margin_kmh = margin_kmh
        self.plates = plate_reader or NullPlateReader()
        self.jpeg_quality = jpeg_quality
        self.edge_px = edge_px    # a box touching the frame edge is cut off, so its centre is wrong
        self._plate_frame = {}   # track_id -> (image, vehicle box, all tracks) of the frame its plate box was last found in
        self._plate_seen = {}
        self.photo_drawer = None   # function(image, tracks, focus_track_id) -> picture, set by the program
        self._best = {}   # track_id -> (box area, image, box): the closest, sharpest view so far
        self._kinds = {}  # track_id -> how often each vehicle kind was seen (car, motorcycle, bus, truck)
        self._seen = {}   # track_id -> last time seen, to free frames of vehicles that left
        self.stats = {"readings": 0, "valid": 0, "violations": 0, "horns": 0}
        self.last_tracks = []
        self.measured = {}   # track id -> last measured speed, for the live picture
        # Horn events: the sound arrives right away but the camera picture lags it, so a horn is
        # matched to the frames received `video_lag_s` after it, once those have arrived.
        self.video_lag_s = video_lag_s
        self.horn_min_box_px = horn_min_box_px
        self.mic_offset_db = mic_offset_db
        # A horn counts only if it was loud enough and long enough (the limits the admin sets on the website),
        # and, with horn_zone set, only if the vehicle that could have made it stands in the measured area.
        self.horn_min_level_db = horn_min_level_db
        self.horn_min_seconds = horn_min_seconds
        self.horn_zone = horn_zone      # function(box, (width, height)) -> True when inside the zone
        self._history = collections.deque()   # (frame time, image, detections) of frames with vehicles
        self._horns = []

    @property
    def flag_above_kmh(self):
        return self.speed_limit_kmh + self.margin_kmh

    def process(self, image, t, wall):
        """Handle one frame. Returns a list of dicts describing violations sent this frame."""
        h, w = image.shape[:2]
        tracks = [d for d in self.tracker.track(image) if d.name in VEHICLE_KINDS]
        tracks = [d for d in tracks if not self._is_parked(d, t)]   # parked vehicles are not tracked at all
        self.last_tracks = tracks
        events = []
        stamp = wall.timestamp()
        self._stamp = stamp
        if tracks:
            self._history.append((stamp, image, list(tracks)))
        while self._history and stamp - self._history[0][0] > self.video_lag_s + 6.0:
            self._history.popleft()
        for d in tracks:
            if self.estimator.is_done(d.track_id):
                continue  # already measured and reported; nothing more to keep
            x1, y1, x2, y2 = d.box
            self._seen[d.track_id] = t
            self._kinds.setdefault(d.track_id, collections.Counter())[d.name] += 1
            area = (x2 - x1) * (y2 - y1)
            if d.track_id not in self._best or area > self._best[d.track_id][0]:
                self._best[d.track_id] = (area, image, d.box, tracks)
            self.plates.observe(d.track_id, image, d.box)
            lb = getattr(self.plates, "last_box", lambda _t: None)(d.track_id)
            if lb is not None and lb != self._plate_seen.get(d.track_id):
                self._plate_seen[d.track_id] = lb      # a new plate box: keep this frame, the box fits it
                self._plate_frame[d.track_id] = (image, d.box, tracks)
            if x1 <= self.edge_px or x2 >= w - self.edge_px or y2 >= h - self.edge_px:
                continue  # vehicle partly out of view: its box moves slower than the vehicle does
            reading = self.estimator.update(d.track_id, t, (x1 + x2) / 2.0, y2, (w, h))
            if reading:
                self._handle(reading, wall, events)
        for reading in self.estimator.expire(t):
            self._handle(reading, wall, events)
        for track_id in [k for k, seen in self._seen.items() if t - seen > 10.0]:
            self._seen.pop(track_id)
            self._release(track_id)
        ready = [h for h in self._horns if stamp >= h.end + self.video_lag_s + 0.3]
        for h in ready:
            self._horns.remove(h)
            ev = self._report_horn(h, wall)
            if ev:
                events.append(ev)
        return events

    def _is_parked(self, d, t):
        """A vehicle that stays in the same place for a while is parked. It gets no box, no plate reading
        and no speed. If it drives off, it counts again from that moment.

        This is judged by place, not by track number: the tracker often gives a parked vehicle a new
        number every few seconds, which would start the clock over each time."""
        if not self.parked_after_s:
            return False
        box = tuple(float(v) for v in d.box)
        if t - self._stills_pruned >= 1.0:
            self._stills = [st for st in self._stills if t - st["last"] <= 5.0]
            self._stills_pruned = t
        best, best_iou = None, 0.0
        for st in self._stills:
            v = _iou(st["box"], box)
            if v > best_iou:
                best, best_iou = st, v
        if best is None or best_iou < 0.6:
            self._stills.append({"box": box, "first": t, "last": t})
            return False
        best["last"] = t
        parked = t - best["first"] >= self.parked_after_s
        if parked:
            self._seen[d.track_id] = t     # still in view, so it is cleaned up only after it leaves
        return parked

    def add_horn(self, horn):
        """A horn was heard. It is reported once the matching video frames have arrived."""
        self._horns.append(horn)

    def flush(self, wall):
        """End of video: report vehicles that were still being measured."""
        events = []
        for reading in self.estimator.finalize_all():
            self._handle(reading, wall, events)
        for h in self._horns:
            ev = self._report_horn(h, wall)
            if ev:
                events.append(ev)
        self._horns = []
        return events

    def _handle(self, reading, wall, events):
        self.stats["readings"] += 1
        if not reading.valid:
            log.debug("Track %s not measured: %s", reading.track_id, reading.reason)
            return
        self.stats["valid"] += 1
        self.measured[reading.track_id] = reading.speed_kmh   # shown on the live picture
        if len(self.measured) > 200:
            self.measured.pop(next(iter(self.measured)))
        log.info("Track %s: %.1f km/h (%d frames, %.1f m)", reading.track_id, reading.speed_kmh,
                 reading.n_samples, reading.distance_m)
        if reading.speed_kmh > self.flag_above_kmh:
            events.append(self._report(reading, wall))
        self._release(reading.track_id)

    def _report(self, reading, wall):
        plate, confidence = self.plates.result(reading.track_id)
        guess, guess_conf = self.plates.guess(reading.track_id)
        fields = {
            "type": "speeding",
            "speed_kmh": f"{reading.speed_kmh:.1f}",
            "detected_at": wall.isoformat(timespec="seconds"),
        }
        kind = self._kind(reading.track_id)
        if kind:
            fields["vehicle_class"] = kind
        if plate:
            fields["plate"] = plate
            if confidence is not None:
                fields["plate_confidence"] = f"{confidence:.3f}"
        if plate:
            note = f"plate {plate}"
        elif guess:
            note = f"plate unclear: {guess}?"   # shown on the photo for the administrator, not sent as a plate
        else:
            note = None
        jpeg = self._evidence(reading, note)
        status = self._deliver(fields, jpeg, self._stamp)
        self.stats["violations"] += 1
        log.warning("Speeding %.1f km/h (limit %g): %s", reading.speed_kmh, self.speed_limit_kmh, status)
        return dict(fields, status=status, track_id=reading.track_id)

    def _report_horn(self, horn, wall):
        """Match a horn to the vehicle in view. Only one vehicle in view means we can say whose it was.
        With several, the record goes to the administrator without a plate, with all of them marked."""
        if horn.duration_s < self.horn_min_seconds:
            log.info("Horn heard (%.1f s) but shorter than the %.1f s that counts. Not reported.",
                     horn.duration_s, self.horn_min_seconds)
            return None
        if self.horn_min_level_db is not None and self.mic_offset_db is not None \
                and horn.level_db + self.mic_offset_db < self.horn_min_level_db:
            log.info("Horn heard at %.0f dB, quieter than the %.0f dB threshold: too far away. Not reported.",
                     horn.level_db + self.mic_offset_db, self.horn_min_level_db)
            return None
        lo, hi = horn.start + self.video_lag_s - 0.3, horn.end + self.video_lag_s + 0.3
        frames = []
        for stamp, image, dets in self._history:
            if lo <= stamp <= hi:
                near = [d for d in dets if d.box[2] - d.box[0] >= self.horn_min_box_px
                        and (self.horn_zone is None or self.horn_zone(d.box, image.shape[1::-1]))]
                if near:
                    frames.append((stamp, image, near))
        ids = sorted({d.track_id for _, _, near in frames for d in near})
        if not ids:
            log.info("Horn heard (%.0f Hz, %.1f s) but no vehicle was in the measured area. Not reported.",
                     horn.peak_hz, horn.duration_s)
            return None
        fields = {"type": "honking", "detected_at": datetime.fromtimestamp(horn.start, wall.tzinfo)
                  .isoformat(timespec="seconds")}
        if self.mic_offset_db is not None:
            fields["db_level"] = f"{min(200.0, max(0.0, horn.level_db + self.mic_offset_db)):.1f}"
        if getattr(horn, "sound_note", None):
            fields["sound_note"] = horn.sound_note
        track_id = ids[0] if len(ids) == 1 else None
        note = f"HORN {horn.duration_s:.1f} s"
        if track_id is not None:
            kinds = collections.Counter(d.name for _, _, near_ in frames for d in near_ if d.track_id == track_id)
            if kinds:
                fields["vehicle_class"] = kinds.most_common(1)[0][0]
            plate, confidence = self.plates.result(track_id)
            guess, _ = self.plates.guess(track_id)
            if plate:
                fields["plate"] = plate
                if confidence is not None:
                    fields["plate_confidence"] = f"{confidence:.3f}"
                note += f"  plate {plate}"
            elif guess:
                note += f"  plate unclear: {guess}?"
        else:
            note += f"  {len(ids)} vehicles in view, horn source unknown"
        # the frame where the vehicles were largest shows them best
        stamp, image, near = max(frames, key=lambda f: max((d.box[2] - d.box[0]) for d in f[2]))
        img = image.copy()
        colour = (0, 0, 255) if track_id is not None else (0, 140, 255)
        for d in near:
            x1, y1, x2, y2 = (int(v) for v in d.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 3)
        x1, y1 = int(min(d.box[0] for d in near)), int(min(d.box[1] for d in near))
        self._label(img, x1, y1, note, colour)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        status = self._deliver(fields, buf.tobytes() if ok else None, horn.start + self.video_lag_s)
        self.stats["horns"] += 1
        log.warning("Horn %.1f s at %.0f Hz, %s: %s", horn.duration_s, horn.peak_hz,
                    f"track {track_id}" if track_id is not None else f"{len(ids)} vehicles in view", status)
        return dict(fields, status=status, track_id=track_id)

    def _deliver(self, fields, jpeg, event_stamp):
        """Send a violation. With clips on, it goes out once its video is cut (a few seconds later)."""
        if self.clips is None or not (self.clips.ready or self.clips.audio_on):
            return self.sender.send_violation(fields, jpeg)

        def go(video, audio):
            status = self.sender.send_violation(fields, jpeg, video=video, audio=audio)
            parts = ([f"a {len(video) // 1024} KB clip"] if video else []) + \
                    ([f"a {len(audio) // 1024} KB sound clip"] if audio else [])
            log.info("Sent %s violation with %s: %s", fields.get("type"),
                     " and ".join(parts) if parts else "the photo only", status)
        self.clips.request(event_stamp, go, with_audio=True)
        return "recording clip"

    @staticmethod
    def _label(img, x, y, text, colour):
        cv2.rectangle(img, (x, max(0, y - 32)), (x + 12 * len(text), y), colour, -1)
        cv2.putText(img, text, (x + 4, max(18, y - 9)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

    def _evidence(self, reading, note=None):
        best = self._best.get(reading.track_id)
        if best is None:
            return None
        _, image, box = best[:3]
        tracks = best[3] if len(best) > 3 else None
        label = f"{reading.speed_kmh:.1f} km/h (limit {self.speed_limit_kmh:g})"
        if note:
            label += f"  {note}"
        drawn = None
        if self.photo_drawer is not None and tracks is not None:
            # Show the vehicle that violated, with its own plate box. The picture comes from the frame where
            # that plate box was found, so the box sits on the plate. Other vehicles are thin boxes only.
            pf = self._plate_frame.get(reading.track_id)
            if pf is not None:
                image, box, tracks = pf
            try:
                drawn = self.photo_drawer(image, tracks, reading.track_id)
            except Exception:  # noqa: BLE001 - fall back to the plain photo
                log.exception("Could not draw the evidence photo")
        if drawn is not None:
            img = drawn
            scale = max(1.0, img.shape[1] / 1280.0)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, max(2, int(2 * scale)))
            cv2.rectangle(img, (0, 0), (tw + 24, th + 24), (0, 0, 200), -1)
            cv2.putText(img, label, (12, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale,
                        (255, 255, 255), max(2, int(2 * scale)), cv2.LINE_AA)
        else:
            img = image.copy()
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            self._label(img, x1, y1, label, (0, 0, 255))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return buf.tobytes() if ok else None

    def _kind(self, track_id):
        seen = self._kinds.get(track_id)
        return seen.most_common(1)[0][0] if seen else None

    def _release(self, track_id):
        self._best.pop(track_id, None)
        self._plate_frame.pop(track_id, None)
        self._plate_seen.pop(track_id, None)
        self._kinds.pop(track_id, None)
        self.plates.forget(track_id)
