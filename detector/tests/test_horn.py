"""Horn detection tests with synthetic sound. Run:  python -m unittest discover -s tests -v

Synthetic sound proves the logic works and that the obvious non-horns are rejected. It does not prove
accuracy with your microphone and your road: measure that with horn_tool.py evaluate."""
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from detector.audio import HornDetector, detect_in_samples, load_wav, save_wav  # noqa: E402
from detector.pipeline import Pipeline  # noqa: E402
from detector.speed import Calibration, SpeedEstimator  # noqa: E402
from detector.tracker import Detection  # noqa: E402

SR = 16000
TZ = timezone(timedelta(hours=8))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


class Sounds:
    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)

    def ambient(self, sec, level=0.02):
        w = np.convolve(self.rng.normal(size=int(sec * SR)), np.ones(8) / 8, "same")
        return (w * level * 3).astype(np.float32)

    def horn(self, sec, f=420.0, ratio=1.25, wobble=0.0):
        t = np.arange(int(sec * SR)) / SR
        drift = 1 + wobble * np.sin(2 * np.pi * 5 * t)
        x = np.zeros_like(t)
        for ff in (f, f * ratio):
            ph = 2 * np.pi * np.cumsum(ff * drift) / SR + self.rng.uniform(0, 6)
            x += sum(np.sin(h * ph) / h for h in (1, 2, 3))
        return x.astype(np.float32)

    def speech(self, sec):
        t = np.arange(int(sec * SR)) / SR
        pitch = 120 + 30 * np.sin(2 * np.pi * 3 * t) + 20 * np.sin(2 * np.pi * 0.7 * t)
        ph = 2 * np.pi * np.cumsum(pitch) / SR
        return (sum(np.sin(h * ph) / h for h in range(1, 25)) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)

    def engine_sweep(self, sec):
        t = np.arange(int(sec * SR)) / SR
        ph = 2 * np.pi * np.cumsum(60 + 20 * np.sin(2 * np.pi * 0.5 * t)) / SR
        return sum(np.sin(h * ph) / h for h in range(1, 12)).astype(np.float32)

    def passing_vehicle(self, sec):
        n = int(sec * SR)
        env = np.exp(-0.5 * ((np.arange(n) - n / 2) / (n / 5)) ** 2)
        return (np.convolve(self.rng.normal(size=n), np.ones(6) / 6, "same") * env).astype(np.float32)

    def clap(self):
        x = np.zeros(800, np.float32)
        x[:20] = self.rng.normal(size=20)
        return x

    @staticmethod
    def mix(background, sound, at, snr_db):
        out = background.copy()
        i = int(at * SR)
        sound = sound * (rms(background) * 10 ** (snr_db / 20) / rms(sound))
        out[i:i + len(sound)] += sound[:len(out) - i]
        return out


class HornDetection(unittest.TestCase):
    def setUp(self):
        self.s = Sounds()

    def test_finds_a_horn_and_times_it(self):
        x = self.s.mix(self.s.ambient(12), self.s.horn(0.6), 6.0, 16)
        events = detect_in_samples(x, SR)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertAlmostEqual(ev.start, 6.0, delta=0.1)
        self.assertGreaterEqual(ev.duration_s, 0.2)
        self.assertAlmostEqual(ev.peak_hz, 420, delta=40)
        self.assertGreater(ev.excess_db, 10)

    def test_horns_over_a_range_of_pitches_and_loudness(self):
        for f in (330, 420, 520):
            for snr in (13, 18, 26):
                x = self.s.mix(self.s.ambient(10), self.s.horn(0.8, f), 5.0, snr)
                self.assertEqual(len(detect_in_samples(x, SR)), 1, f"{f} Hz at {snr} dB")

    def test_short_beep(self):
        x = self.s.mix(self.s.ambient(10), self.s.horn(0.25), 5.0, 18)
        self.assertEqual(len(detect_in_samples(x, SR)), 1)

    def test_slightly_unsteady_horn(self):
        x = self.s.mix(self.s.ambient(10), self.s.horn(0.8, wobble=0.01), 5.0, 18)
        self.assertEqual(len(detect_in_samples(x, SR)), 1)

    def test_works_on_a_loud_and_on_a_quiet_road(self):
        for level in (0.002, 0.2):
            x = self.s.mix(self.s.ambient(10, level), self.s.horn(0.6), 5.0, 16)
            self.assertEqual(len(detect_in_samples(x, SR)), 1, f"background {level}")

    def test_a_horn_barely_above_the_background_is_not_detected(self):
        x = self.s.mix(self.s.ambient(10), self.s.horn(0.6), 5.0, 5)
        self.assertEqual(detect_in_samples(x, SR), [])

    def test_a_few_beeps_in_a_row_are_one_event(self):
        x = self.s.ambient(14)
        for at in (5.0, 5.5, 6.0):
            x = self.s.mix(x, self.s.horn(0.25), at, 18)
        self.assertEqual(len(detect_in_samples(x, SR)), 1)

    def test_two_horns_far_apart_are_two_events(self):
        x = self.s.mix(self.s.ambient(20), self.s.horn(0.5), 5.0, 16)
        x = self.s.mix(x, self.s.horn(0.5), 12.0, 16)
        self.assertEqual(len(detect_in_samples(x, SR)), 2)

    def test_a_long_horn_is_reported_once(self):
        x = self.s.mix(self.s.ambient(14), self.s.horn(4.0), 5.0, 16)
        self.assertEqual(len(detect_in_samples(x, SR)), 1)

    def test_sounds_that_are_not_horns(self):
        cases = {"speech": self.s.speech(2.5), "engine whine": self.s.engine_sweep(2.5),
                 "passing vehicle": self.s.passing_vehicle(4), "clap": self.s.clap()}
        for name, sound in cases.items():
            for snr in (12, 20):
                x = self.s.mix(self.s.ambient(12), sound, 5.0, snr)
                self.assertEqual(detect_in_samples(x, SR), [], f"{name} at {snr} dB")

    def test_silence_and_noise_alone(self):
        self.assertEqual(detect_in_samples(np.zeros(SR * 10, np.float32), SR), [])
        self.assertEqual(detect_in_samples(self.s.ambient(30), SR), [])

    def test_a_steady_tone_that_never_stops_is_not_reported_forever(self):
        # a generator whine starts and runs for a minute: one event at most, then it is background
        x = self.s.mix(self.s.ambient(70), self.s.horn(60.0), 5.0, 16)
        self.assertLessEqual(len(detect_in_samples(x, SR)), 1)

    def test_result_does_not_depend_on_how_the_audio_is_chunked(self):
        x = self.s.mix(self.s.ambient(12), self.s.horn(0.6), 6.0, 16)
        reference = detect_in_samples(x, SR, chunk_s=0.1)
        for step in (333, 1000, 4096):
            det, got = HornDetector(SR), []
            for i in range(0, len(x), step):
                got += det.feed(x[i:i + step], i / SR)
            self.assertEqual(len(got), len(reference), step)
            self.assertAlmostEqual(got[0].start, reference[0].start, delta=0.05)

    def test_other_sample_rates(self):
        for sr in (22050, 44100):
            n = int(sr * 10)
            t = np.arange(n) / sr
            rng = np.random.default_rng(3)
            x = (np.convolve(rng.normal(size=n), np.ones(8) / 8, "same") * 0.06).astype(np.float32)
            horn = sum(np.sin(2 * np.pi * f * h * t[:int(sr * 0.6)]) / h for f in (420, 525) for h in (1, 2, 3))
            x[int(sr * 5):int(sr * 5) + len(horn)] += (horn * 0.3).astype(np.float32)
            self.assertEqual(len(detect_in_samples(x, sr)), 1, sr)

    def test_a_gap_in_the_audio_stream_resets_cleanly(self):
        det = HornDetector(SR)
        bg = self.s.ambient(3)
        det.feed(bg, 0.0)
        # microphone dropped out for 5 s, then a horn arrives
        x = self.s.mix(self.s.ambient(6), self.s.horn(0.6), 4.0, 16)
        events = det.feed(x, 8.0)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].start, 12.0, delta=0.15)

    def test_wav_round_trip(self):
        x = self.s.mix(self.s.ambient(6), self.s.horn(0.6), 3.0, 16)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.wav")
            save_wav(path, x, SR)
            y, sr = load_wav(path)
        self.assertEqual(sr, SR)
        self.assertEqual(len(detect_in_samples(y, sr)), 1)


class FakeTracker:
    def __init__(self, script):
        self.script, self.i = script, 0

    def track(self, image):
        d = self.script[self.i] if self.i < len(self.script) else []
        self.i += 1
        return d


class FakeSender:
    def __init__(self):
        self.sent = []

    def send_violation(self, fields, jpeg=None):
        self.sent.append((fields, jpeg))
        return "sent"


class FakePlates:
    def observe(self, *a):
        pass

    def result(self, track_id):
        return ("ABC1234", 0.9) if track_id == 1 else (None, None)

    def guess(self, track_id):
        return ("XYZ9876", 0.4) if track_id == 3 else (None, None)

    def forget(self, track_id):
        pass


class HornInThePipeline(unittest.TestCase):
    FPS = 10.0
    T0 = 1_800_000_000.0

    def run_scene(self, vehicles_by_frame, horn_start, lag=1.0, seconds=8.0, **pipe_kwargs):
        """vehicles_by_frame(frame_index) -> list of Detection. Horn heard at horn_start s after T0.
        The camera picture runs `lag` seconds behind the sound, as with a real stream."""
        cal = Calibration.from_rectangle([(300, 650), (980, 650), (800, 300), (480, 300)], 3.0, 8.0, (1280, 720))
        n = int(seconds * self.FPS)
        sender = FakeSender()
        pipe = Pipeline(FakeTracker([vehicles_by_frame(i) for i in range(n)]), SpeedEstimator(cal), sender,
                        10.0, 0.0, plate_reader=FakePlates(), video_lag_s=lag, mic_offset_db=100.0, **pipe_kwargs)
        from detector.audio import HornEvent
        horn = HornEvent(start=self.T0 + horn_start, end=self.T0 + horn_start + 0.4, duration_s=0.4,
                         peak_hz=420.0, level_db=-12.0, excess_db=20.0, tonal=0.95)
        image = np.full((720, 1280, 3), 60, np.uint8)
        events = []
        for i in range(n):
            t = i / self.FPS
            wall = datetime.fromtimestamp(self.T0 + t, TZ)
            # the picture received at wall time W shows what happened `lag` seconds earlier
            if abs(t - (horn_start + 0.4)) < 0.5 / self.FPS:
                pipe.add_horn(horn)
            events += pipe.process(image, t, wall)
        events += pipe.flush(datetime.fromtimestamp(self.T0 + seconds, TZ))
        return sender, events

    @staticmethod
    def car(track_id, x=600, w=200):
        return Detection(track_id, (x, 300, x + w, 500), 0.9, 2, "car")

    def test_horn_gates_loudness_duration_and_zone(self):
        scene = lambda i: [self.car(1)] if 10 <= i <= 40 else []
        # quiet horn (level -12 dB + 100 offset = 88 dB) against a 95 dB threshold: far away, not reported
        sender, _ = self.run_scene(scene, 2.0, horn_min_level_db=95.0)
        self.assertEqual(len(sender.sent), 0)
        sender, _ = self.run_scene(scene, 2.0, horn_min_level_db=80.0)
        self.assertEqual(len(sender.sent), 1)
        # a 0.4 s horn is shorter than the 1 s that counts
        sender, _ = self.run_scene(scene, 2.0, horn_min_seconds=1.0)
        self.assertEqual(len(sender.sent), 0)
        # the vehicle is outside the measured area: not blamed
        sender, _ = self.run_scene(scene, 2.0, horn_zone=lambda box, size: False)
        self.assertEqual(len(sender.sent), 0)
        sender, _ = self.run_scene(scene, 2.0, horn_zone=lambda box, size: True)
        self.assertEqual(len(sender.sent), 1)

    def test_horn_from_the_only_vehicle_in_view_is_linked_to_its_plate(self):
        # the car is in view in the frames that show the moment of the horn (1 s later in the picture)
        sender, events = self.run_scene(lambda i: [self.car(1)] if 10 <= i <= 40 else [], horn_start=2.0)
        self.assertEqual(len(sender.sent), 1)
        fields, jpeg = sender.sent[0]
        self.assertEqual(fields["type"], "honking")
        self.assertEqual(fields["plate"], "ABC1234")
        self.assertEqual(fields["db_level"], "88.0")
        self.assertTrue(fields["detected_at"].startswith("2027-01-15T"))
        self.assertEqual(jpeg[:3], b"\xff\xd8\xff")

    def test_the_camera_lag_decides_which_vehicle_is_matched(self):
        # car 1 passes only before the horn is visible, car 2 only when the picture catches up
        def scene(i):
            if i < 10:
                return [self.car(1)]
            if 30 <= i <= 40:
                return [self.car(2)]
            return []
        sender, _ = self.run_scene(scene, horn_start=2.0, lag=1.0)
        self.assertEqual(len(sender.sent), 1)
        self.assertNotIn("plate", sender.sent[0][0])        # car 2 has no plate; car 1 is not matched
        self.assertNotIn("ABC1234", str(sender.sent[0][0]))

    def test_unclear_plate_is_shown_on_the_photo_but_not_sent(self):
        sender, _ = self.run_scene(lambda i: [self.car(3)] if 10 <= i <= 40 else [], horn_start=2.0)
        self.assertEqual(len(sender.sent), 1)
        self.assertNotIn("plate", sender.sent[0][0])

    def test_two_vehicles_in_view_means_no_plate_is_assigned(self):
        sender, _ = self.run_scene(lambda i: [self.car(1, 300), self.car(2, 800)] if 10 <= i <= 40 else [],
                                   horn_start=2.0)
        self.assertEqual(len(sender.sent), 1)
        fields, jpeg = sender.sent[0]
        self.assertEqual(fields["type"], "honking")
        self.assertNotIn("plate", fields)
        self.assertIsNotNone(jpeg)

    def test_horn_with_no_vehicle_in_view_is_not_reported(self):
        sender, events = self.run_scene(lambda i: [], horn_start=2.0)
        self.assertEqual(sender.sent, [])
        self.assertEqual(events, [])

    def test_vehicle_too_far_away_to_be_the_source_is_ignored(self):
        sender, _ = self.run_scene(lambda i: [self.car(1, w=40)] if 10 <= i <= 40 else [], horn_start=2.0)
        self.assertEqual(sender.sent, [])

    def test_horn_at_the_very_end_of_the_video_is_still_reported_on_flush(self):
        sender, _ = self.run_scene(lambda i: [self.car(1)] if i >= 60 else [], horn_start=6.5, seconds=8.0)
        self.assertEqual(len(sender.sent), 1)

    def test_history_does_not_grow_without_limit(self):
        cal = Calibration.from_rectangle([(300, 650), (980, 650), (800, 300), (480, 300)], 3.0, 8.0, (1280, 720))
        pipe = Pipeline(FakeTracker([[self.car(1)]] * 2000), SpeedEstimator(cal), FakeSender(), 10.0, 0.0,
                        video_lag_s=1.0)
        image = np.zeros((720, 1280, 3), np.uint8)
        for i in range(2000):
            pipe.process(image, i / 10.0, datetime.fromtimestamp(self.T0 + i / 10.0, TZ))
        self.assertLess(len(pipe._history), 100)


class ToolTests(unittest.TestCase):
    def test_evaluate_counts_hits_and_false_alarms(self):
        s = Sounds(9)
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "horn"))
            os.makedirs(os.path.join(d, "other"))
            for k in range(3):
                save_wav(os.path.join(d, "horn", f"h{k}.wav"), s.mix(s.ambient(8), s.horn(0.6), 4.0, 16), SR)
            save_wav(os.path.join(d, "horn", "quiet.wav"), s.mix(s.ambient(8), s.horn(0.6), 4.0, 3), SR)
            for k in range(2):
                save_wav(os.path.join(d, "other", f"o{k}.wav"), s.mix(s.ambient(8), s.speech(2), 3.0, 14), SR)
            out = subprocess.run([sys.executable, os.path.join(ROOT, "horn_tool.py"), "evaluate",
                                  "--horn", os.path.join(d, "horn"), "--other", os.path.join(d, "other")],
                                 capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("3 of 4 detected", out.stdout)
        self.assertIn("0 false alarm", out.stdout)
        self.assertIn("quiet.wav", out.stdout)


if __name__ == "__main__":
    unittest.main()
