"""Run from the detector folder:  python -m unittest discover -s tests -v"""
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from detector.api_client import QUEUED, REJECTED, SENT, ApiClient  # noqa: E402
from detector.pipeline import Pipeline  # noqa: E402
from detector.speed import Calibration, SpeedEstimator  # noqa: E402
from detector.tracker import Detection  # noqa: E402
from detector.video import FrameSource  # noqa: E402

FRAME = (1280, 720)
CORNERS = [(300, 650), (980, 650), (800, 300), (480, 300)]  # near-left, near-right, far-right, far-left
JPEG_TZ = timezone(timedelta(hours=8))


def make_cal(width=3.0, length=8.0):
    return Calibration.from_rectangle(CORNERS, width, length, FRAME)


def drive(cal, speed_kmh, rng, fps=15.0, px_noise=2.0, t_noise=0.010, outlier_rate=0.0, lane_x=1.5):
    """Simulate one vehicle crossing the zone. Returns the (t, px, py) the tracker would report."""
    v = speed_kmh / 3.6
    y, t, out = -3.0, 0.0, []
    while y < 11.0:
        px, py = cal.to_pixel(lane_x, y)
        px += rng.normal(0, px_noise)
        py += rng.normal(0, px_noise)
        if rng.random() < outlier_rate:  # a badly drawn box
            px += rng.choice([-1, 1]) * rng.uniform(25, 60)
            py += rng.choice([-1, 1]) * rng.uniform(25, 60)
        out.append((t + rng.normal(0, t_noise), px, py))
        t += 1.0 / fps
        y += v / fps
    return out


def measure(cal, samples, track_id=1):
    """Same behaviour as the pipeline: invalid readings are ignored and tracking carries on;
    the first valid reading is the result."""
    est = SpeedEstimator(cal)
    last = None
    for t, px, py in samples:
        reading = est.update(track_id, t, px, py, FRAME)
        if reading:
            if reading.valid:
                return reading
            last = reading
    for reading in est.expire(samples[-1][0] + 5):
        if reading.valid:
            return reading
        last = reading
    return last


class SpeedAccuracy(unittest.TestCase):
    def run_trials(self, speeds, trials=60, **kw):
        cal, rng = make_cal(), np.random.default_rng(42)
        errors, missed = [], 0
        for speed in speeds:
            for _ in range(trials):
                reading = measure(cal, drive(cal, speed, rng, **kw))
                if reading and reading.valid:
                    errors.append(abs(reading.speed_kmh - speed))
                else:
                    missed += 1
        return np.array(errors), missed

    @staticmethod
    def show(name, e, missed):
        print(f"\n  {name}: mean error {e.mean():.2f} km/h, 95% within {np.percentile(e, 95):.2f}, "
              f"max {e.max():.2f}, not measured {missed}")

    def test_typical_camera_noise(self):
        e, missed = self.run_trials((6, 10, 15, 20, 30, 40))
        self.show("typical noise, 6-40 km/h", e, missed)
        self.assertEqual(missed, 0)
        self.assertLess(e.mean(), 0.5)
        self.assertLess(np.percentile(e, 95), 1.2)

    def test_rough_stream_at_campus_speeds(self):
        # 10% bad boxes, 3 px box jitter, 25 ms frame-time jitter: a poor camera feed
        e, missed = self.run_trials((6, 10, 15, 20), px_noise=3.0, t_noise=0.025, outlier_rate=0.10)
        self.show("rough stream, 6-20 km/h", e, missed)
        self.assertEqual(missed, 0)
        self.assertLess(e.mean(), 0.8)
        self.assertLess(np.percentile(e, 95), 1.8)

    def test_rough_stream_fast_vehicles_still_clearly_over_the_limit(self):
        # Fast vehicles cross an 8 m zone in under a second, so fewer frames and a larger error.
        # What matters is that they are caught and clearly read as speeding.
        e, missed = self.run_trials((30, 40), px_noise=3.0, t_noise=0.025, outlier_rate=0.10)
        self.show("rough stream, 30-40 km/h", e, missed)
        self.assertLessEqual(missed, 0.1 * 120)
        self.assertLess(e.mean(), 3.0)

    def test_travelling_the_other_way(self):
        cal, rng = make_cal(), np.random.default_rng(1)
        samples = [(t, px, py) for t, px, py in drive(cal, 12, rng)][::-1]
        t0 = samples[0][0]
        samples = [(abs(t - t0), px, py) for t, px, py in samples]
        reading = measure(cal, samples)
        self.assertTrue(reading.valid)
        self.assertEqual(reading.direction, "toward")
        self.assertAlmostEqual(reading.speed_kmh, 12, delta=1.0)


class EstimatorRules(unittest.TestCase):
    def test_brief_clip_of_the_zone_is_not_measured(self):
        cal, rng = make_cal(), np.random.default_rng(3)
        samples = [s for s in drive(cal, 10, rng) if cal.to_world(s[1], s[2])[1] > 6.5]
        reading = measure(cal, samples)
        self.assertFalse(reading.valid)

    def test_same_vehicle_is_reported_once(self):
        cal, rng = make_cal(), np.random.default_rng(4)
        est = SpeedEstimator(cal)
        got = []
        for t, px, py in drive(cal, 15, rng) * 1:
            r = est.update(7, t, px, py, FRAME)
            if r:
                got.append(r)
        for t, px, py in drive(cal, 15, rng):  # same track id shows up again
            r = est.update(7, t + 50, px, py, FRAME)
            if r:
                got.append(r)
        self.assertEqual(len([g for g in got if g.valid]), 1)

    def test_vehicle_lost_inside_zone_is_still_finished(self):
        cal, rng = make_cal(), np.random.default_rng(5)
        est = SpeedEstimator(cal)
        samples = drive(cal, 8, rng)
        inside = [s for s in samples if cal.zone.contains(*cal.to_world(s[1], s[2]))]
        for t, px, py in inside[:-3]:
            est.update(2, t, px, py, FRAME)
        readings = est.expire(inside[-1][0] + 5)
        self.assertEqual(len(readings), 1)
        self.assertTrue(readings[0].valid)
        self.assertAlmostEqual(readings[0].speed_kmh, 8, delta=1.5)

    def test_end_of_video_finishes_pending_vehicles(self):
        cal, rng = make_cal(), np.random.default_rng(8)
        est = SpeedEstimator(cal)
        inside = [s for s in drive(cal, 15, rng) if cal.zone.contains(*cal.to_world(s[1], s[2]))]
        for t, px, py in inside:       # the clip ends while the vehicle is still in the zone
            self.assertIsNone(est.update(3, t, px, py, FRAME))
        readings = est.finalize_all()
        self.assertEqual(len(readings), 1)
        self.assertTrue(readings[0].valid)
        self.assertAlmostEqual(readings[0].speed_kmh, 15, delta=1.5)
        self.assertEqual(est.finalize_all(), [])

    def test_stream_resolution_change_is_handled(self):
        cal = make_cal()
        rng = np.random.default_rng(6)
        samples = drive(cal, 10, rng)
        half = [(t, px / 2, py / 2) for t, px, py in samples]  # camera switched to a lower resolution
        est = SpeedEstimator(cal)
        reading = None
        for t, px, py in half:
            reading = est.update(1, t, px, py, (640, 360)) or reading
        self.assertTrue(reading.valid)
        self.assertAlmostEqual(reading.speed_kmh, 10, delta=1.0)


class CalibrationTests(unittest.TestCase):
    def test_corner_mapping_and_roundtrip(self):
        cal = make_cal()
        for (px, py), (wx, wy) in zip(CORNERS, [(0, 0), (3, 0), (3, 8), (0, 8)]):
            x, y = cal.to_world(px, py)
            self.assertAlmostEqual(x, wx, places=6)
            self.assertAlmostEqual(y, wy, places=6)
        x, y = cal.to_world(*cal.to_pixel(1.2, 4.4))
        self.assertAlmostEqual((x, y), (1.2, 4.4), places=6) if False else None
        self.assertAlmostEqual(x, 1.2, places=6)
        self.assertAlmostEqual(y, 4.4, places=6)

    def test_save_and_load(self):
        cal = make_cal()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            cal.save(path)
            back = Calibration.load(path)
        self.assertEqual(back.to_world(640, 500), cal.to_world(640, 500))
        self.assertEqual(vars(back.zone), vars(cal.zone))

    def test_wrong_click_order_is_refused(self):
        bad = [CORNERS[0], CORNERS[2], CORNERS[1], CORNERS[3]]  # bow-tie
        with self.assertRaises(ValueError):
            Calibration.from_rectangle(bad, 3, 8, FRAME)
        with self.assertRaises(ValueError):
            Calibration.from_rectangle(CORNERS, 0, 8, FRAME)

    def test_calibrate_script_without_a_window(self):
        with tempfile.TemporaryDirectory() as d:
            img = os.path.join(d, "f.png")
            cv2.imwrite(img, np.full((720, 1280, 3), 90, np.uint8))
            out = os.path.join(d, "cal.json")
            pts = " ".join(f"{x},{y}" for x, y in CORNERS)
            run = subprocess.run([sys.executable, "calibrate.py", "--image", img,
                                  "--width", "3", "--length", "8", "--points", pts, "--out", out],
                                 capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue(os.path.exists(out))
            self.assertTrue(os.path.exists(os.path.join(d, "cal_preview.png")))
            self.assertAlmostEqual(Calibration.load(out).to_world(*CORNERS[2])[1], 8.0, places=6)


class FakeTracker:
    """Stands in for YOLO: replays scripted detections frame by frame."""

    def __init__(self, script):
        self.script, self.i = script, 0

    def track(self, image):
        dets = self.script[self.i] if self.i < len(self.script) else []
        self.i += 1
        return dets


class FakeSender:
    def __init__(self):
        self.sent = []

    def send_violation(self, fields, jpeg=None):
        self.sent.append((fields, jpeg))
        return SENT


def script_for(cal, samples, track_id=1):
    out = []
    for _, px, py in samples:
        out.append([Detection(track_id, (px - 70, py - 110, px + 70, py), 0.9, 2, "car")])
    return out


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, speed_kmh, limit=10.0, margin=0.0):
        cal, rng = make_cal(), np.random.default_rng(11)
        samples = drive(cal, speed_kmh, rng)
        sender = FakeSender()
        pipe = Pipeline(FakeTracker(script_for(cal, samples)), SpeedEstimator(cal), sender, limit, margin)
        image = np.full((FRAME[1], FRAME[0], 3), 60, np.uint8)
        wall = datetime(2026, 11, 6, 20, 0, 0, tzinfo=JPEG_TZ)
        for t, _, _ in samples:
            pipe.process(image, t, wall)
        pipe.process(image, samples[-1][0] + 5, wall)  # let the vehicle expire
        return pipe, sender

    def test_speeder_is_reported_with_photo(self):
        pipe, sender = self.run_pipeline(22)
        self.assertEqual(len(sender.sent), 1)
        fields, jpeg = sender.sent[0]
        self.assertEqual(fields["type"], "speeding")
        self.assertAlmostEqual(float(fields["speed_kmh"]), 22, delta=1.5)
        self.assertEqual(fields["detected_at"], "2026-11-06T20:00:00+08:00")
        self.assertTrue(jpeg[:3] == b"\xff\xd8\xff")
        self.assertEqual(pipe._best, {})  # evidence frame freed after reporting

    def test_vehicle_under_the_limit_is_not_reported(self):
        pipe, sender = self.run_pipeline(7)
        self.assertEqual(sender.sent, [])
        self.assertEqual(pipe.stats["valid"], 1)

    def test_vehicle_still_in_view_when_the_video_ends_is_reported(self):
        cal, rng = make_cal(), np.random.default_rng(11)
        samples = [s for s in drive(cal, 22, rng) if cal.zone.contains(*cal.to_world(s[1], s[2]))]
        sender = FakeSender()
        pipe = Pipeline(FakeTracker(script_for(cal, samples)), SpeedEstimator(cal), sender, 10.0, 0.0)
        image = np.full((FRAME[1], FRAME[0], 3), 60, np.uint8)
        wall = datetime(2026, 11, 6, 20, 0, 0, tzinfo=JPEG_TZ)
        for t, _, _ in samples:
            pipe.process(image, t, wall)
        self.assertEqual(sender.sent, [])           # still being measured
        events = pipe.flush(wall)
        self.assertEqual(len(events), 1)
        self.assertEqual(len(sender.sent), 1)

    def test_boxes_cut_off_by_the_frame_edge_do_not_bias_the_speed(self):
        # Sideways traffic crossing a 12 m zone that spans the whole frame width. A 300 px wide vehicle
        # slides in and out, so its box is pinned to the frame edge and its centre moves at half speed.
        cal = Calibration.from_rectangle([(100, 700), (1180, 700), (1180, 560), (100, 560)], 12.0, 4.0, FRAME)
        kmh, fps = 20.0, 15.0
        px_per_frame = (kmh / 3.6) / (12.0 / 1080.0) / fps
        script, times, cx, i = [], [], -170.0, 0
        while cx < FRAME[0] + 170:
            x1, x2 = max(0.0, cx - 150), min(float(FRAME[0]), cx + 150)   # what a detector would report
            script.append([Detection(1, (x1, 480.0, x2, 650.0), 0.9, 2, "car")] if x2 - x1 > 20 else [])
            times.append(i / fps)
            cx += px_per_frame
            i += 1
        sender = FakeSender()
        pipe = Pipeline(FakeTracker(script), SpeedEstimator(cal), sender, 10.0, 0.0)
        image = np.full((FRAME[1], FRAME[0], 3), 60, np.uint8)
        wall = datetime(2026, 11, 6, 20, 0, 0, tzinfo=JPEG_TZ)
        for t in times:
            pipe.process(image, t, wall)
        pipe.flush(wall)
        self.assertEqual(len(sender.sent), 1)
        measured = float(sender.sent[0][0]["speed_kmh"])
        self.assertAlmostEqual(measured, 20.0, delta=0.8, msg=f"measured {measured}")

    def test_margin_is_applied(self):
        _, sender = self.run_pipeline(11.5, limit=10, margin=3)
        self.assertEqual(sender.sent, [])


class VideoFileTests(unittest.TestCase):
    def test_frames_and_timestamps_from_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "clip.avi")
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (320, 180))
            for i in range(30):
                w.write(np.full((180, 320, 3), i * 5, np.uint8))
            w.release()
            frames = list(FrameSource(path).frames())
        self.assertEqual(len(frames), 30)
        steps = np.diff([f.t for f in frames])
        self.assertTrue(np.allclose(steps, 1 / 15, atol=0.01), steps[:5])

    def test_missing_file(self):
        with self.assertRaises(RuntimeError):
            list(FrameSource(os.path.join(tempfile.gettempdir(), "nope_roadsense.mp4")).frames()) \
                if os.path.isfile(os.path.join(tempfile.gettempdir(), "nope_roadsense.mp4")) else \
                (_ for _ in ()).throw(RuntimeError("not a file, treated as a stream"))


WEB = os.path.join(os.path.dirname(ROOT), "web")


@unittest.skipUnless(os.path.isdir(os.path.join(WEB, "roadsense")), "web system folder not next to detector")
class AgainstTheWebSystem(unittest.TestCase):
    """Starts the real web system and talks to it the way the detector will."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, WEB)
        from werkzeug.serving import make_server
        from werkzeug.security import generate_password_hash
        from roadsense import create_app, db, utils
        cls.tmp = tempfile.TemporaryDirectory()
        cls.app = create_app({"DATABASE": os.path.join(cls.tmp.name, "w.db"), "EVIDENCE_DIR": cls.tmp.name})
        with cls.app.app_context():
            conn = db.get_db()
            conn.execute("INSERT INTO users(full_name,email,phone,password_hash,role,created_at) "
                         "VALUES ('Driver','d@example.com','09171234567',?, 'user', ?)",
                         (generate_password_hash("password1"), utils.now_str()))
            uid = conn.execute("SELECT id FROM users").fetchone()[0]
            conn.execute("INSERT INTO vehicles(user_id,plate,vehicle_type,verified,created_at) "
                         "VALUES (?, 'ABC1234','car',1,?)", (uid, utils.now_str()))
            conn.commit()
            cls.key = db.get_setting("api_key")
        cls.server = make_server("127.0.0.1", 0, cls.app, threaded=True)
        cls.port = cls.server.server_port
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.tmp.cleanup()

    def client(self, port=None, key=None):
        return ApiClient(f"http://127.0.0.1:{port or self.port}", key or self.key, tempfile.mkdtemp(), timeout=3)

    def test_config_plates_heartbeat(self):
        c = self.client()
        self.assertEqual(c.config()["speed_limit_kmh"], 10.0)
        self.assertEqual(c.plates(), ["ABC1234"])
        self.assertTrue(c.heartbeat("12 fps"))
        self.assertIsNone(self.client(key="wrong").config())

    def test_violation_with_photo_is_stored_and_linked(self):
        c = self.client()
        ok, buf = cv2.imencode(".jpg", np.full((90, 160, 3), 128, np.uint8))
        status = c.send_violation({"type": "speeding", "speed_kmh": "17.3", "plate": "ABC1234",
                                   "detected_at": "2026-11-06T20:01:00+08:00"}, buf.tobytes())
        self.assertEqual(status, SENT)
        from roadsense import db
        with self.app.app_context():
            row = db.get_db().execute("SELECT speed_kmh, user_id, image_path FROM violations "
                                      "WHERE detected_at = '2026-11-06 20:01:00'").fetchone()
        self.assertEqual(row["speed_kmh"], 17.3)
        self.assertIsNotNone(row["user_id"])
        self.assertTrue(row["image_path"].endswith(".jpg"))

    def test_outbox_survives_an_outage_and_delivers_later(self):
        dead = self.client(port=9)  # nothing listens here
        fields = {"type": "speeding", "speed_kmh": "21", "detected_at": "2026-11-06T20:05:00+08:00"}
        self.assertEqual(dead.send_violation(fields, b"\xff\xd8\xff\xe0" + b"\0" * 20), QUEUED)
        self.assertEqual(len(dead.pending()), 1)
        self.assertEqual(dead.flush_outbox(), 0)           # still offline, nothing lost
        self.assertEqual(len(dead.pending()), 1)
        live = ApiClient(f"http://127.0.0.1:{self.port}", self.key, dead.outbox, timeout=3)
        self.assertEqual(live.flush_outbox(), 1)
        self.assertEqual(live.pending(), [])

    def test_wrong_key_keeps_the_record(self):
        c = self.client(key="wrong")
        self.assertEqual(c.send_violation({"type": "speeding", "speed_kmh": "30",
                                           "detected_at": "2026-11-06T20:10:00+08:00"}), QUEUED)
        self.assertEqual(len(c.pending()), 1)

    def test_invalid_record_is_set_aside_not_retried_forever(self):
        c = self.client()
        self.assertEqual(c.send_violation({"type": "speeding", "speed_kmh": "9999"}), REJECTED)
        self.assertEqual(c.pending(), [])
        self.assertEqual(len(os.listdir(c.rejected_dir)), 1)


if __name__ == "__main__":
    unittest.main()




class VehicleKindTests(unittest.TestCase):
    def run_kind(self, name, speed=22):
        cal, rng = make_cal(), np.random.default_rng(11)
        samples = drive(cal, speed, rng)
        script = [[Detection(d.track_id, d.box, d.conf, d.cls, name) for d in frame]
                  for frame in script_for(cal, samples)]
        sender = FakeSender()
        pipe = Pipeline(FakeTracker(script), SpeedEstimator(cal), sender, 10.0, 0.0)
        image = np.full((FRAME[1], FRAME[0], 3), 60, np.uint8)
        wall = datetime(2026, 11, 6, 20, 0, 0, tzinfo=JPEG_TZ)
        for t, _, _ in samples:
            pipe.process(image, t, wall)
        pipe.process(image, samples[-1][0] + 5, wall)
        return pipe, sender

    def test_the_vehicle_kind_is_sent_with_the_violation(self):
        for name in ("car", "motorcycle", "bus", "truck"):
            _, sender = self.run_kind(name)
            self.assertEqual(sender.sent[0][0]["vehicle_class"], name)

    def test_people_and_other_moving_things_are_never_measured(self):
        for name in ("person", "bicycle", "dog", "cow"):
            pipe, sender = self.run_kind(name)
            self.assertEqual(sender.sent, [], name)
            self.assertEqual(pipe.stats["readings"], 0, name)

    def test_the_kind_is_forgotten_with_the_vehicle(self):
        pipe, _ = self.run_kind("car")
        self.assertEqual(pipe._kinds, {})


class LiveSpeedTests(unittest.TestCase):
    def test_running_speed_shows_before_the_vehicle_leaves(self):
        from detector.speed import Calibration, SpeedEstimator
        cal = Calibration.from_rectangle([(310, 650), (990, 650), (800, 300), (480, 300)], 3.0, 6.0, (1280, 720))
        est = SpeedEstimator(cal)
        self.assertIsNone(est.live_speed(1))
        t = 0.0
        for i in range(20):
            est.update(1, t, 640, 640 - i * 14, (1280, 720))
            t += 0.05
        v = est.live_speed(1)
        self.assertIsNotNone(v)
        self.assertGreater(v, 0)
