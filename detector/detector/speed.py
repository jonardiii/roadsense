"""Vehicle speed from a ground-plane homography.

You mark a rectangle of known size on the road (width x length in metres) and click its four
corners in a camera frame. That gives a mapping from image pixels to road coordinates in
metres, so a vehicle's position is tracked in real distance instead of pixels.

Speed is the robust (Theil-Sen, median-of-slopes) fit of position against frame time over the
frames where the vehicle is inside the marked zone. A few bad bounding boxes or a late frame
do not move the median, which is what cheap, compressed camera streams need.

Road coordinates: X is across the road, Y runs along it (the length of your rectangle).
"""
import json
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

MPS_TO_KMH = 3.6


@dataclass
class Zone:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x, y):
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


class Calibration:
    def __init__(self, image_points, world_points, frame_size, zone=None, lateral_margin=1.0):
        self.image_points = np.asarray(image_points, dtype=float).reshape(4, 2)
        self.world_points = np.asarray(world_points, dtype=float).reshape(4, 2)
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self._check_corner_order()
        H, _ = cv2.findHomography(self.image_points, self.world_points, 0)
        if H is None:
            raise ValueError("The four corners don't define a usable road area. Click them again.")
        self.H = H
        self.H_inv = np.linalg.inv(H)
        wp = self.world_points
        self.zone = zone or Zone(wp[:, 0].min() - lateral_margin, wp[:, 0].max() + lateral_margin,
                                 wp[:, 1].min(), wp[:, 1].max())

    @classmethod
    def from_rectangle(cls, image_points, width_m, length_m, frame_size):
        """Corners in order: near-left, near-right, far-right, far-left (as the camera sees them)."""
        if width_m <= 0 or length_m <= 0:
            raise ValueError("Width and length must be greater than zero.")
        world = [(0, 0), (width_m, 0), (width_m, length_m), (0, length_m)]
        return cls(image_points, world, frame_size)

    def _check_corner_order(self):
        p = self.image_points
        cross = []
        for i in range(4):
            a, b, c = p[i], p[(i + 1) % 4], p[(i + 2) % 4]
            cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
            raise ValueError("The corners don't form a four-sided shape in order. Click near-left, "
                             "near-right, far-right, far-left.")

    # ---- mapping ----
    def to_world(self, px, py, frame_size=None):
        if frame_size and tuple(frame_size) != self.frame_size:
            px *= self.frame_size[0] / frame_size[0]
            py *= self.frame_size[1] / frame_size[1]
        v = self.H @ np.array([px, py, 1.0])
        if abs(v[2]) < 1e-9:
            return None
        return float(v[0] / v[2]), float(v[1] / v[2])

    def to_pixel(self, x, y):
        v = self.H_inv @ np.array([x, y, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    # ---- files ----
    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "image_points": self.image_points.tolist(),
                "world_points": self.world_points.tolist(),
                "frame_size": list(self.frame_size),
                "zone": vars(self.zone),
            }, fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(d["image_points"], d["world_points"], d["frame_size"], zone=Zone(**d["zone"]))

    # ---- drawing ----
    def draw(self, image, grid_m=1.0):
        """Overlay the measurement zone and a metre grid so you can check the calibration by eye."""
        out = image.copy()
        z = self.zone
        corners = [(z.x_min, z.y_min), (z.x_max, z.y_min), (z.x_max, z.y_max), (z.x_min, z.y_max)]
        poly = np.array([self.to_pixel(*c) for c in corners], dtype=np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [poly], (0, 200, 255))
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
        cv2.polylines(out, [poly], True, (0, 220, 255), 2, cv2.LINE_AA)
        y = z.y_min
        while y <= z.y_max + 1e-6:
            a = tuple(int(v) for v in self.to_pixel(z.x_min, y))
            b = tuple(int(v) for v in self.to_pixel(z.x_max, y))
            cv2.line(out, a, b, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(out, f"{y:g} m", (b[0] + 6, b[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            y += grid_m
        for p in self.image_points:
            cv2.circle(out, (int(p[0]), int(p[1])), 7, (0, 0, 255), -1, cv2.LINE_AA)
        return out


@dataclass
class SpeedReading:
    track_id: int
    valid: bool
    reason: str
    speed_kmh: float = 0.0
    direction: str = ""
    n_samples: int = 0
    duration_s: float = 0.0
    distance_m: float = 0.0
    residual_m: float = 0.0


def _theil_sen(t, v):
    """Median of all pairwise slopes of v against t, plus the matching intercept."""
    if len(t) > 150:  # keep the pair count small
        keep = np.linspace(0, len(t) - 1, 150).astype(int)
        t, v = t[keep], v[keep]
    i, j = np.triu_indices(len(t), k=1)
    dt = t[j] - t[i]
    ok = dt > 1e-3
    slope = float(np.median((v[j][ok] - v[i][ok]) / dt[ok]))
    intercept = float(np.median(v - slope * t))
    return slope, intercept


class SpeedEstimator:
    def __init__(self, calibration, min_samples=6, min_duration_s=0.4, min_distance_m=2.0,
                 max_residual_m=0.5, max_speed_kmh=80.0, max_gap_s=1.0, exit_frames=3,
                 timing_slack_s=0.03):
        self.cal = calibration
        # Frame times wobble a little, and a faster vehicle moves further in that wobble, so the
        # scatter we accept grows with speed: max_residual_m + timing_slack_s * speed (m/s).
        self.timing_slack_s = timing_slack_s
        self.min_samples = min_samples
        self.min_duration_s = min_duration_s
        self.min_distance_m = min_distance_m
        self.max_residual_m = max_residual_m
        self.max_speed_kmh = max_speed_kmh
        self.max_gap_s = max_gap_s
        self.exit_frames = exit_frames   # frames in a row outside the zone before a vehicle counts as gone
        self._samples = {}
        self._out_streak = {}
        self._last_seen = {}
        self._done = set()
        self._done_order = deque(maxlen=1000)

    def is_done(self, track_id):
        return track_id in self._done

    def live_speed(self, track_id):
        """Running speed while a vehicle is still inside the zone, for the live picture only.
        It skips the checks that decide whether a reading is trusted. None until enough frames."""
        samples = self._samples.get(track_id)
        if not samples or len(samples) < self.min_samples:
            return None
        a = np.array(samples, dtype=float)
        t = a[:, 0] - a[0, 0]
        if t[-1] - t[0] < self.min_duration_s:
            return None
        vx, _ = _theil_sen(t, a[:, 1])
        vy, _ = _theil_sen(t, a[:, 2])
        return float(np.hypot(vx, vy)) * MPS_TO_KMH

    def update(self, track_id, t, px, py, frame_size=None):
        """Feed one tracked position (the bottom-centre of the vehicle box). Returns a
        SpeedReading when the vehicle leaves the zone, otherwise None."""
        if track_id in self._done:
            return None
        pos = self.cal.to_world(px, py, frame_size)
        if pos is None:
            return None
        self._last_seen[track_id] = t
        if self.cal.zone.contains(*pos):
            self._out_streak[track_id] = 0
            self._samples.setdefault(track_id, []).append((t, pos[0], pos[1]))
            return None
        if self._samples.get(track_id):
            # One stray box outside the zone must not end the measurement; a real exit
            # stays outside for several frames.
            self._out_streak[track_id] = self._out_streak.get(track_id, 0) + 1
            if self._out_streak[track_id] >= self.exit_frames:
                return self.finalize(track_id)
        return None

    def expire(self, now):
        """Finish vehicles that vanished (tracking lost) while still inside the zone."""
        stale = [k for k, seen in self._last_seen.items() if now - seen > self.max_gap_s]
        readings = []
        for track_id in stale:
            if self._samples.get(track_id):
                readings.append(self.finalize(track_id))
            self._last_seen.pop(track_id, None)
            self._out_streak.pop(track_id, None)
        return readings

    def finalize_all(self):
        """Finish every vehicle still being measured. Call when the video or stream ends."""
        readings = [self.finalize(k) for k in list(self._samples) if self._samples[k]]
        self._last_seen.clear()
        self._out_streak.clear()
        return readings

    def finalize(self, track_id):
        samples = self._samples.pop(track_id, [])
        self._out_streak.pop(track_id, None)
        reading = self._measure(track_id, samples)
        if reading.valid:
            self._mark_done(track_id)
        return reading

    def _mark_done(self, track_id):
        if len(self._done_order) == self._done_order.maxlen:
            self._done.discard(self._done_order[0])
        self._done_order.append(track_id)
        self._done.add(track_id)

    def _measure(self, track_id, samples):
        n = len(samples)
        if n < self.min_samples:
            return SpeedReading(track_id, False, "too_few_frames_in_zone", n_samples=n)
        a = np.array(samples, dtype=float)
        t, x, y = a[:, 0] - a[0, 0], a[:, 1], a[:, 2]
        duration = float(t[-1] - t[0])
        if duration < self.min_duration_s:
            return SpeedReading(track_id, False, "too_short_in_time", n_samples=n, duration_s=duration)
        vx, bx = _theil_sen(t, x)
        vy, by = _theil_sen(t, y)
        speed = float(np.hypot(vx, vy)) * MPS_TO_KMH
        distance = speed / MPS_TO_KMH * duration
        resid = float(np.median(np.hypot(x - (vx * t + bx), y - (vy * t + by))))
        base = dict(n_samples=n, duration_s=duration, distance_m=distance, residual_m=resid,
                    speed_kmh=speed, direction="away" if vy > 0 else "toward")
        if distance < self.min_distance_m:
            return SpeedReading(track_id, False, "too_short_in_distance", **base)
        if resid > self.max_residual_m + self.timing_slack_s * speed / MPS_TO_KMH:
            return SpeedReading(track_id, False, "unsteady_track", **base)
        if speed > self.max_speed_kmh:
            return SpeedReading(track_id, False, "implausible_speed", **base)
        return SpeedReading(track_id, True, "ok", **base)
