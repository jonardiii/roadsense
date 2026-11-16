"""Plate reading checks. Run from the detector folder:  python -m unittest discover -s tests -v"""
import os
import random
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from detector.pipeline import Pipeline  # noqa: E402
from detector.plates import formats  # noqa: E402
from detector.plates.ocr import ALLOWED, FastPlateOcr, join_results, make_ocr  # noqa: E402
from detector.plates.reader import PlateReader, plausible_shape, read_best  # noqa: E402
from detector.plates.voting import PlateVote  # noqa: E402
from detector.speed import SpeedEstimator  # noqa: E402
from test_detector import FRAME, FakeSender, FakeTracker, JPEG_TZ, drive, make_cal, script_for  # noqa: E402
from datetime import datetime  # noqa: E402

SWAPS = {"B": "8", "8": "B", "O": "0", "0": "O", "I": "1", "1": "I", "S": "5", "5": "S", "Z": "2", "2": "Z",
         "G": "6", "6": "G"}


class FormatTests(unittest.TestCase):
    def test_valid_plates_are_left_alone(self):
        for plate in ("ABC1234", "ABC123", "AB12345", "AB1234", "1234AB", "123ABC"):
            self.assertEqual(formats.correct(plate), (plate, False))
            self.assertTrue(formats.is_valid_format(plate))

    def test_clean_removes_spaces_and_dashes(self):
        self.assertEqual(formats.clean(" abc-1234 "), "ABC1234")

    def test_letter_digit_confusions_are_fixed(self):
        self.assertEqual(formats.correct("ABC1Z34"), ("ABC1234", True))   # Z in a digit slot
        self.assertEqual(formats.correct("A8C1234"), ("ABC1234", True))   # 8 in a letter slot
        self.assertEqual(formats.correct("ABCI234"), ("ABC1234", True))   # I in a digit slot
        self.assertEqual(formats.correct("A8C1Z34"), ("ABC1234", True))   # two at once

    def test_ambiguous_text_is_not_guessed(self):
        # six characters that two different formats explain equally well
        text, changed = formats.correct("AB12CD")
        self.assertFalse(changed and text != "AB12CD" and not formats.is_valid_format(text))
        # nothing maps to a digit here, so no format fits
        self.assertEqual(formats.correct("ABCXYZ9"), ("ABCXYZ9", False))

    def test_unknown_length_untouched(self):
        self.assertEqual(formats.correct("AB1"), ("AB1", False))
        self.assertFalse(formats.is_valid_format("ABCD1234"))


def random_plate(rng):
    return "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(3)) + "".join(rng.choice("0123456789") for _ in range(4))


def noisy_read(true, rng, p):
    """One OCR reading with independent character errors (an optimistic error model)."""
    out, errors = [], 0
    for ch in true:
        if rng.random() < p:
            errors += 1
            out.append(SWAPS[ch] if ch in SWAPS and rng.random() < 0.5 else rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ0123456789"))
        else:
            out.append(ch)
    conf = rng.uniform(0.5, 0.85) if errors else rng.uniform(0.75, 0.97)
    return "".join(out), conf


class VotingTests(unittest.TestCase):
    def evaluate(self, p, n_reads, trials=600, seed=5):
        rng = random.Random(seed)
        single_ok = voted_ok = accepted = wrong_accepted = 0
        for _ in range(trials):
            true = random_plate(rng)
            reads = [noisy_read(true, rng, p) for _ in range(n_reads)]
            fixed, _ = formats.correct(reads[0][0])
            single_ok += fixed == true
            vote = PlateVote()
            for text, conf in reads:
                vote.add(text, conf, quality=rng.uniform(0.5, 1.0))
            c = vote.consensus()
            voted_ok += c.text == true
            if c.confidence >= 0.6:
                accepted += 1
                wrong_accepted += c.text != true
        return single_ok / trials, voted_ok / trials, accepted / trials, wrong_accepted / max(accepted, 1)

    def test_voting_beats_a_single_frame(self):
        print("\n  Simulated OCR with independent character errors (best case; real errors repeat more):")
        for p in (0.05, 0.10, 0.20):
            single, voted, accepted, wrong = self.evaluate(p, n_reads=12)
            print(f"    {p:.0%} of characters misread per frame: one frame {single:.0%} correct, "
                  f"vote over 12 frames {voted:.0%}; sent to the website {accepted:.0%} of the time, "
                  f"{wrong:.1%} of those wrong")
            self.assertGreater(voted, single)
        single, voted, accepted, wrong = self.evaluate(0.10, n_reads=12)
        self.assertGreater(voted, 0.95)
        self.assertLess(wrong, 0.02)

    def test_few_frames_lower_the_confidence(self):
        a, b = PlateVote(), PlateVote()
        a.add("ABC1234", 0.9)
        for _ in range(6):
            b.add("ABC1234", 0.9)
        self.assertLess(a.consensus().confidence, b.consensus().confidence)
        self.assertLess(a.consensus().confidence, 0.7)

    def test_disagreeing_frames_lower_the_confidence(self):
        agree, split = PlateVote(), PlateVote()
        rng = random.Random(1)
        for _ in range(8):
            agree.add("ABC1234", 0.9)
            split.add("".join(rng.choice("ABCDEFGH") for _ in range(3)) + "".join(rng.choice("0123456789") for _ in range(4)), 0.9)
        self.assertGreater(agree.consensus().confidence, 0.8)
        self.assertLess(split.consensus().confidence, 0.5)

    def test_unusable_text_is_ignored(self):
        v = PlateVote()
        self.assertFalse(v.add("A1", 0.9))
        self.assertFalse(v.add("ABCDEFGHIJKL", 0.9))
        self.assertIsNone(v.consensus())

    def test_weak_readings_are_dropped_first(self):
        v = PlateVote(max_reads=3)
        for conf in (0.9, 0.8, 0.7, 0.95):
            v.add("ABC1234", conf)
        self.assertEqual(len(v), 3)
        self.assertNotIn(0.7, [c for _, c, _ in v.reads])


class SnapToRegisteredPlates(unittest.TestCase):
    def test_exact_match_gets_a_small_boost(self):
        v = PlateVote()
        for _ in range(5):
            v.add("ABC1234", 0.8)
        plain = v.consensus().confidence
        known = v.consensus(["ABC1234"])
        self.assertTrue(known.verified)
        self.assertGreater(known.confidence, plain)

    def test_one_character_off_snaps_when_a_frame_saw_it(self):
        v = PlateVote()
        for text in ["ABD1234"] * 5 + ["ABC1234"] * 2:       # two frames saw the C
            v.add(text, 0.85)
        self.assertEqual(v.consensus().text, "ABD1234")
        c = v.consensus(["ABC1234"])
        self.assertEqual(c.text, "ABC1234")
        self.assertTrue(c.snapped)

    def test_no_snap_without_evidence(self):
        v = PlateVote()
        for _ in range(6):
            v.add("ABD1234", 0.9)                            # no frame ever saw a C
        c = v.consensus(["ABC1234"])
        self.assertEqual(c.text, "ABD1234")
        self.assertFalse(c.snapped)

    def test_no_snap_when_two_registered_plates_fit(self):
        v = PlateVote()
        for text in ["ABD1234"] * 4 + ["ABC1234"] * 2 + ["ABE1234"] * 2:
            v.add(text, 0.85)
        c = v.consensus(["ABC1234", "ABE1234"])
        self.assertFalse(c.snapped)
        self.assertEqual(c.text, "ABD1234")

    def test_systematic_misreads_are_a_known_limit(self):
        """If every frame makes the same mistake, voting can't see it. Only the registered list helps."""
        rng = random.Random(3)
        v = PlateVote()
        for _ in range(12):
            v.add("ABD1234" if rng.random() < 0.6 else "ABC1234", 0.9)   # 60% of frames misread C as D
        without = v.consensus()
        with_list = v.consensus(["ABC1234"])
        print(f"\n  60% of frames misread one letter: without the registered list the answer is "
              f"{without.text} ({without.confidence:.2f}); with it, {with_list.text}")
        self.assertEqual(with_list.text, "ABC1234")


class FakeDetector:
    def __init__(self, plate_w=160):
        self.plate_w, self.calls = plate_w, 0

    def detect(self, region):
        self.calls += 1
        return (10, 20, 10 + self.plate_w, 20 + self.plate_w // 3), 0.9


class FakeOcr:
    def __init__(self, texts):
        self.texts, self.calls = list(texts), 0

    def read(self, plate):
        self.calls += 1
        text, conf = self.texts[(self.calls - 1) % len(self.texts)]
        return text, conf


def sharp_image(seed=0):
    return np.random.default_rng(seed).integers(0, 255, (FRAME[1], FRAME[0], 3), dtype=np.uint8)


class ReaderTests(unittest.TestCase):
    BOX = (200, 100, 600, 500)

    def run_frames(self, reader, n=24, image=None, box=None):
        image = sharp_image() if image is None else image
        for _ in range(n):
            reader.observe(1, image, box or self.BOX)

    def test_reads_every_nth_frame_and_votes(self):
        ocr = FakeOcr([("ABC1234", 0.9)])
        r = PlateReader(FakeDetector(), ocr, every_n_frames=2)
        self.run_frames(r, 24)
        self.assertEqual(ocr.calls, 12)
        plate, conf = r.result(1)
        self.assertEqual(plate, "ABC1234")
        self.assertGreater(conf, 0.8)

    def test_far_away_vehicles_are_not_read(self):
        ocr = FakeOcr([("ABC1234", 0.9)])
        r = PlateReader(FakeDetector(), ocr, min_vehicle_width=120)
        self.run_frames(r, 20, box=(200, 100, 280, 180))            # 80 px wide
        self.assertEqual(ocr.calls, 0)
        self.assertEqual(r.result(1), (None, None))

    def test_tiny_plates_are_skipped(self):
        ocr = FakeOcr([("ABC1234", 0.9)])
        r = PlateReader(FakeDetector(plate_w=30), ocr, min_plate_width=50)
        self.run_frames(r, 20)
        self.assertEqual(ocr.calls, 0)

    def test_blurry_frames_are_skipped(self):
        ocr = FakeOcr([("ABC1234", 0.9)])
        r = PlateReader(FakeDetector(), ocr, min_sharpness=25)
        self.run_frames(r, 20, image=np.full((FRAME[1], FRAME[0], 3), 90, np.uint8))
        self.assertEqual(ocr.calls, 0)

    def test_work_per_vehicle_is_capped(self):
        ocr = FakeOcr([("ABC1234", 0.9)])
        r = PlateReader(FakeDetector(), ocr, every_n_frames=1, max_attempts=10)
        self.run_frames(r, 100)
        self.assertEqual(ocr.calls, 10)

    def test_unclear_plate_is_withheld_but_remembered(self):
        rng = random.Random(9)
        junk = [("".join(rng.choice("ABCDEFGH") for _ in range(3)) + "".join(rng.choice("0123456789") for _ in range(4)), 0.55)
                for _ in range(12)]
        r = PlateReader(FakeDetector(), FakeOcr(junk), every_n_frames=1, min_confidence=0.6)
        self.run_frames(r, 12)
        plate, conf = r.result(1)
        self.assertIsNone(plate)
        self.assertIsNotNone(conf)
        self.assertIsNotNone(r.guess(1)[0])

    def test_text_that_is_not_a_plate_format_is_not_used(self):
        r = PlateReader(FakeDetector(), FakeOcr([("FORMULA1", 0.95)]), every_n_frames=1, require_valid_format=True)
        self.run_frames(r, 12)
        plate, conf = r.result(1)
        self.assertIsNone(plate)
        self.assertIsNone(r.visible(1))
        loose = PlateReader(FakeDetector(), FakeOcr([("FORMULA1", 0.95)]), every_n_frames=1)
        self.run_frames(loose, 12)
        self.assertIsNotNone(loose.result(1)[0])

    def test_real_plate_format_is_used_and_drawn(self):
        r = PlateReader(FakeDetector(), FakeOcr([("KAU8922", 0.95)]), every_n_frames=1, require_valid_format=True)
        self.run_frames(r, 12)
        self.assertEqual(r.result(1)[0], "KAU8922")
        self.assertIsNotNone(r.visible(1))

    def test_forget_frees_the_vehicle(self):
        r = PlateReader(FakeDetector(), FakeOcr([("ABC1234", 0.9)]))
        self.run_frames(r, 6)
        r.forget(1)
        self.assertEqual(r.guess(1), (None, None))


class PipelineWithPlates(unittest.TestCase):
    def run_pipeline(self, ocr_texts, verified=()):
        cal = make_cal()
        samples = drive(cal, 22, np.random.default_rng(11))
        sender = FakeSender()
        reader = PlateReader(FakeDetector(plate_w=80), FakeOcr(ocr_texts), verified, every_n_frames=1,
                             min_vehicle_width=100, min_plate_width=50, min_sharpness=5)
        pipe = Pipeline(FakeTracker(script_for(cal, samples)), SpeedEstimator(cal), sender, 10.0, 0.0,
                        plate_reader=reader)
        image = sharp_image()
        wall = datetime(2026, 11, 7, 10, 0, 0, tzinfo=JPEG_TZ)
        for t, _, _ in samples:
            pipe.process(image, t, wall)
        pipe.process(image, samples[-1][0] + 5, wall)
        return sender

    def test_confident_plate_is_sent_with_the_violation(self):
        sender = self.run_pipeline([("ABC1234", 0.92)])
        fields, jpeg = sender.sent[0]
        self.assertEqual(fields["plate"], "ABC1234")
        self.assertGreater(float(fields["plate_confidence"]), 0.8)
        self.assertEqual(jpeg[:3], b"\xff\xd8\xff")

    def test_unclear_plate_is_not_sent_but_the_violation_still_is(self):
        rng = random.Random(4)
        junk = [("".join(rng.choice("ABCDEFGH") for _ in range(3)) + "".join(rng.choice("0123456789") for _ in range(4)), 0.5)
                for _ in range(40)]
        sender = self.run_pipeline(junk)
        self.assertEqual(len(sender.sent), 1)
        fields, jpeg = sender.sent[0]
        self.assertNotIn("plate", fields)
        self.assertIsNotNone(jpeg)


class EdgeStrokes(unittest.TestCase):
    """The plate frame is often read as an extra I or 1 at either end (seen on real photos)."""

    def test_strokes_at_both_ends_are_dropped(self):
        self.assertEqual(formats.correct("ILAN31381"), ("LAN3138", True))

    def test_one_stray_character_at_the_end_or_start(self):
        self.assertEqual(formats.correct("CBC20807"), ("CBC2080", True))
        self.assertEqual(formats.correct("1ABC1234"), ("ABC1234", True))
        self.assertEqual(formats.correct("ABC1234I"), ("ABC1234", True))

    def test_a_valid_plate_is_never_trimmed(self):
        for plate in ("LEN918", "MAN4684", "ABC1234", "NAB123"):
            self.assertEqual(formats.correct(plate), (plate, False))

    def test_only_stroke_like_characters_are_dropped(self):
        # a real letter at the end is not removed just to make the text fit
        self.assertEqual(formats.correct("ABC1234K"), ("ABC1234K", False))

    def test_two_possible_plates_means_no_guess(self):
        # dropping the first character gives EKX837, dropping the last gives 7EKX83: both are valid
        # formats, so which one is right is a coin toss and the reading is left alone
        self.assertEqual(formats.correct("7EKX837"), ("7EKX837", False))


class FakeRecognizer:
    def __init__(self, plate, probs):
        self.plate, self.probs, self.seen = plate, probs, None

    def run(self, image, return_confidence=False):
        self.seen = image
        return [type("Pred", (), {"plate": self.plate, "char_probs": self.probs})()]


class FastPlateOcrTests(unittest.TestCase):
    def test_reads_text_and_confidence_and_converts_colours(self):
        import numpy as np
        rec = FakeRecognizer("ABC1234___", [0.9] * 7 + [0.1] * 3)
        crop = np.zeros((20, 60, 3), np.uint8)
        crop[..., 0] = 255                     # pure blue in BGR
        text, conf = FastPlateOcr(recognizer=rec).read(crop)
        self.assertEqual(text, "ABC1234")      # padding characters dropped
        self.assertAlmostEqual(conf, 0.9)      # only the real characters count
        self.assertEqual(int(rec.seen[0, 0, 2]), 255)   # the model gets RGB: blue is now the last channel

    def test_nothing_read_returns_none(self):
        import numpy as np
        self.assertIsNone(FastPlateOcr(recognizer=FakeRecognizer("___", [0.5] * 3)).read(np.zeros((10, 30, 3), np.uint8)))

    def test_engine_choice(self):
        with self.assertRaises(ValueError):
            make_ocr("nonsense")


class OcrHelpers(unittest.TestCase):
    @staticmethod
    def box(x, y, w=40, h=30):
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    def test_single_line_is_read_left_to_right(self):
        res = [(self.box(120, 10), "1234", 0.9), (self.box(10, 12), "ABC", 0.8)]
        text, conf = join_results(res)
        self.assertEqual(text, "ABC1234")
        self.assertAlmostEqual(conf, (0.9 * 4 + 0.8 * 3) / 7)

    def test_two_line_plate_reads_top_line_first(self):
        res = [(self.box(10, 80), "1234", 0.9), (self.box(10, 10), "AB", 0.9)]
        self.assertEqual(join_results(res)[0], "AB1234")

    def test_small_print_on_the_plate_is_not_part_of_the_number(self):
        # big characters plus the small PHILIPPINES banner above and a year sticker beside them
        res = [(self.box(10, 10, w=120, h=12), "PHILIPPINES", 0.9),
               (self.box(10, 30, w=60, h=40), "ABC", 0.9), (self.box(80, 30, w=70, h=40), "1234", 0.9),
               (self.box(160, 30, w=25, h=14), "2019", 0.8)]
        text, conf = join_results(res)
        self.assertEqual(text, "ABC1234")
        self.assertAlmostEqual(conf, 0.9)

    def test_two_line_plate_keeps_both_lines(self):
        res = [(self.box(10, 80, h=34), "1234", 0.9), (self.box(10, 10, h=36), "AB", 0.9)]
        self.assertEqual(join_results(res)[0], "AB1234")

    def test_nothing_found(self):
        self.assertIsNone(join_results([]))


class RealOcrSmokeTest(unittest.TestCase):
    """Runs the real OCR engine on a drawn plate. Checks the wiring, not real-world accuracy."""

    def test_reads_a_drawn_plate(self):
        try:
            from detector.plates.ocr import EasyOcr
            ocr = EasyOcr(gpu=False)
        except Exception as exc:  # not installed, or the model files can't be downloaded
            self.skipTest(f"EasyOCR not available: {exc}")
        img = np.full((130, 380, 3), 255, np.uint8)
        cv2.putText(img, "ABC 1234", (18, 92), cv2.FONT_HERSHEY_DUPLEX, 2.5, (25, 25, 25), 5, cv2.LINE_AA)
        out = ocr.read(img)
        print(f"\n  OCR read a drawn 'ABC 1234' as: {out}")
        self.assertIsNotNone(out)
        text, conf = out
        self.assertTrue(set(text) <= set(ALLOWED))
        self.assertGreaterEqual(sum(a == b for a, b in zip(text, "ABC1234")), 5)


class DatasetTools(unittest.TestCase):
    def run_tool(self, *args):
        run = subprocess.run([sys.executable, "prepare_dataset.py", *args], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(run.returncode, 0, run.stderr)
        return run.stdout

    def test_voc_conversion_and_merge(self):
        with tempfile.TemporaryDirectory() as d:
            images, ann = os.path.join(d, "kaggle", "images"), os.path.join(d, "kaggle", "annotations")
            os.makedirs(images)
            os.makedirs(ann)
            for i in range(6):
                cv2.imwrite(os.path.join(images, f"car{i}.png"), np.full((100, 200, 3), 120, np.uint8))
                with open(os.path.join(ann, f"car{i}.xml"), "w") as fh:
                    fh.write("<annotation><size><width>200</width><height>100</height></size><object>"
                             "<bndbox><xmin>50</xmin><ymin>20</ymin><xmax>150</xmax><ymax>60</ymax></bndbox>"
                             "</object></annotation>")
            out = os.path.join(d, "kaggle_yolo")
            self.run_tool("voc", "--images", images, "--annotations", ann, "--out", out)
            first = sorted(os.listdir(os.path.join(out, "labels", "train")))[0]
            with open(os.path.join(out, "labels", "train", first)) as fh:
                label = fh.read().split()
            self.assertEqual(label[0], "0")
            for got, want in zip(label[1:], (0.5, 0.4, 0.5, 0.4)):
                self.assertAlmostEqual(float(got), want, places=5)
            n_train = len(os.listdir(os.path.join(out, "images", "train")))
            n_val = len(os.listdir(os.path.join(out, "images", "val")))
            self.assertEqual(n_train + n_val, 6)
            self.assertGreaterEqual(n_val, 1)

            # a Roboflow-style export: train/images + train/labels, other class ids, one polygon label
            rf = os.path.join(d, "rf")
            for split in ("train", "valid"):
                os.makedirs(os.path.join(rf, split, "images"))
                os.makedirs(os.path.join(rf, split, "labels"))
            for split, count in (("train", 2), ("valid", 1)):
                for i in range(count):
                    cv2.imwrite(os.path.join(rf, split, "images", f"p{i}.jpg"), np.full((100, 200, 3), 80, np.uint8))
                    row = "3 0.5 0.5 0.2 0.1" if i == 0 else "7 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6"
                    with open(os.path.join(rf, split, "labels", f"p{i}.txt"), "w") as fh:
                        fh.write(row + "\n")

            merged = os.path.join(d, "merged")
            self.run_tool("merge", "--inputs", out, rf, "--repeats", "1", "3", "--out", merged)
            n_tr = len(os.listdir(os.path.join(merged, "images", "train")))
            n_va = len(os.listdir(os.path.join(merged, "images", "val")))
            self.assertEqual(n_tr, n_train + 2 * 3)      # own set repeated 3 times
            self.assertEqual(n_va, n_val + 1)
            train_labels = os.path.join(merged, "labels", "train")
            for name in os.listdir(train_labels):
                with open(os.path.join(train_labels, name)) as fh:
                    rows = [r.split() for r in fh.read().splitlines() if r.strip()]
                self.assertTrue(all(r[0] == "0" and len(r) == 5 for r in rows), name)
            with open(os.path.join(train_labels, "d1_p1.txt")) as fh:      # the polygon label
                poly = fh.read().split()
            for got, want in zip(poly[1:], (0.5, 0.5, 0.2, 0.2)):          # became a box
                self.assertAlmostEqual(float(got), want, places=5)
            with open(os.path.join(merged, "data.yaml")) as fh:
                self.assertIn("nc: 1", fh.read())

    def test_extract_frames_skips_blurry_ones(self):
        with tempfile.TemporaryDirectory() as d:
            video = os.path.join(d, "clip.avi")
            w = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (320, 180))
            rng = np.random.default_rng(0)
            for i in range(20):
                frame = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8) if i % 2 == 0 else np.full((180, 320, 3), 100, np.uint8)
                w.write(frame)
            w.release()
            out = os.path.join(d, "frames")
            run = subprocess.run([sys.executable, "extract_frames.py", "--video", video, "--out", out, "--every", "1"],
                                 capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(len(os.listdir(out)), 10)   # only the detailed frames


if __name__ == "__main__":
    unittest.main()


class CandidateDetector:
    """Returns several boxes, most confident first, like a detector that rates a muffler above the plate."""
    def __init__(self, boxes):
        self.boxes = boxes

    def candidates(self, image, max_n=3):
        return self.boxes[:max_n]


class SequenceOcr:
    def __init__(self, texts):
        self.texts, self.calls = list(texts), 0

    def read(self, crop):
        text = self.texts[self.calls] if self.calls < len(self.texts) else ""
        self.calls += 1
        return (text, 0.9) if text else None


class WrongObjectTests(unittest.TestCase):
    IMG = np.full((300, 600, 3), 128, np.uint8)
    MUFFLER, PLATE = ((10, 10, 110, 60), 0.9), ((200, 100, 400, 200), 0.7)

    def test_next_candidate_wins_when_the_first_is_not_a_plate(self):
        ocr = SequenceOcr(["XQ", "ABC1234"])
        item = read_best(CandidateDetector([self.MUFFLER, self.PLATE]), ocr, self.IMG)
        self.assertEqual(item["text"], "ABC1234")
        self.assertEqual(item["box"], self.PLATE[0])

    def test_stops_at_the_first_valid_plate(self):
        ocr = SequenceOcr(["ABC1234", "DEF5678"])
        item = read_best(CandidateDetector([self.PLATE, self.MUFFLER]), ocr, self.IMG)
        self.assertEqual(item["text"], "ABC1234")
        self.assertEqual(ocr.calls, 1)

    def test_falls_back_to_the_most_confident_when_none_is_valid(self):
        ocr = SequenceOcr(["XQ", "ZZ"])
        item = read_best(CandidateDetector([self.MUFFLER, self.PLATE]), ocr, self.IMG)
        self.assertEqual(item["box"], self.MUFFLER[0])
        self.assertFalse(item["valid"])

    def test_candidate_limit_is_respected(self):
        ocr = SequenceOcr(["XQ", "ABC1234"])
        item = read_best(CandidateDetector([self.MUFFLER, self.PLATE]), ocr, self.IMG, max_n=1)
        self.assertFalse(item["valid"])
        self.assertEqual(ocr.calls, 1)

    def test_detectors_with_only_detect_still_work(self):
        item = read_best(FakeDetector(), FakeOcr([("ABC1234", 0.9)]), sharp_image())
        self.assertEqual(item["text"], "ABC1234")

    def test_nothing_found_returns_none(self):
        self.assertIsNone(read_best(CandidateDetector([]), FakeOcr([("ABC1234", 0.9)]), self.IMG))

    def test_rejected_candidates_are_not_read(self):
        ocr = SequenceOcr(["ABC1234"])
        item = read_best(CandidateDetector([self.MUFFLER, self.PLATE]), ocr, self.IMG,
                         keep=lambda crop, box: box == self.PLATE[0])
        self.assertEqual(item["box"], self.PLATE[0])
        self.assertEqual(ocr.calls, 1)

    def test_plate_shapes(self):
        self.assertTrue(plausible_shape((0, 0, 200, 100)))      # car plate
        self.assertTrue(plausible_shape((0, 0, 130, 100)))      # motorcycle plate
        self.assertFalse(plausible_shape((0, 0, 100, 100)))     # round sticker
        self.assertFalse(plausible_shape((0, 0, 400, 40)))      # long thin pipe
        self.assertFalse(plausible_shape((0, 0, 0, 10)))

    def test_reader_uses_the_second_candidate_in_the_vehicle_pipeline(self):
        det = CandidateDetector([((10, 10, 110, 60), 0.9), ((200, 100, 400, 200), 0.7)])
        ocr = SequenceOcr(["XQ", "ABC1234"] * 20)
        r = PlateReader(det, ocr, every_n_frames=1, min_plate_width=50, min_sharpness=0)
        for _ in range(4):
            r.observe(1, sharp_image(), (200, 100, 600, 500))
        self.assertEqual(r.guess(1)[0], "ABC1234")


class PhotoNames(unittest.TestCase):
    def test_names_give_the_expected_plate(self):
        from plate_test import expected_plate
        for name, want in [("JAC 7183.jpg", "JAC7183"), ("JAC7183_2.jpg", "JAC7183"), ("NAB 1234 (2).png", "NAB1234"),
                           ("JAC-7183 (3).jpg", "JAC7183"), ("1201-401686.jpg", "1201401686"), ("holiday.jpg", None)]:
            self.assertEqual(expected_plate(name), want, name)


class TemporaryPlate(unittest.TestCase):
    def test_ten_digit_temporary_plate_is_valid(self):
        self.assertTrue(formats.is_valid_format("1201401686"))

    def test_nine_digits_is_not_guessed(self):
        self.assertFalse(formats.is_valid_format("120140168"))
        self.assertEqual(formats.correct("120140168")[0], "120140168")

    def test_letters_in_a_number_plate_are_fixed(self):
        self.assertEqual(formats.correct("12O1401686")[0], "1201401686")


class MotionBlurTool(unittest.TestCase):
    def test_blur_smears_sideways_only(self):
        from distance_test import motion_blur
        img = np.zeros((40, 40, 3), np.uint8)
        img[:, 20] = 255                       # a thin vertical line
        out = motion_blur(img, 9)
        self.assertLess(out[20, 20, 0], 60)     # the line is smeared out
        self.assertGreater(out[20, 17, 0], 0)   # ...into its horizontal neighbours
        img2 = np.zeros((40, 40, 3), np.uint8)
        img2[20, :] = 255                       # a thin horizontal line
        self.assertGreater(motion_blur(img2, 9)[20, 20, 0], 200)   # unchanged

    def test_no_blur_below_two_pixels(self):
        from distance_test import motion_blur
        img = np.random.default_rng(0).integers(0, 255, (10, 10, 3), dtype=np.uint8)
        self.assertTrue((motion_blur(img, 1) == img).all())
