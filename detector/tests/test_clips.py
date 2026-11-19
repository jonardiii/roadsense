import os
import tempfile
import unittest

import numpy as np

from detector.clips import ClipRecorder


class ClipTests(unittest.TestCase):
    def frames(self, rec, seconds, start=1000.0, fps=15):
        for k in range(int(seconds * fps)):
            img = np.full((540, 960, 3), k % 255, np.uint8)
            rec.add(img, start + k / fps)

    def test_clip_is_made_with_frames_before_and_after(self):
        rec = ClipRecorder(before_s=3, after_s=2)
        if not rec.ready:
            self.skipTest("ffmpeg not available")
        got = []
        self.frames(rec, 5)                 # 5 s of history
        rec.request(1000.0 + 4.9, got.append)
        self.assertEqual(got, [])           # still waiting for the 2 s after
        self.frames(rec, 4, start=1005.0)
        rec.close()
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0] and got[0][4:8] == b"ftyp")

    def test_close_finishes_waiting_clip_and_no_ffmpeg_gives_none(self):
        rec = ClipRecorder(before_s=3, after_s=5)
        got = []
        self.frames(rec, 3)
        rec.request(1002.9, got.append)
        rec.close()
        self.assertEqual(len(got), 1)       # delivered even though the after-time never came
        rec2 = ClipRecorder()
        rec2.exe = None
        out = []
        rec2.request(5.0, out.append)
        self.assertEqual(out, [None])


class OutboxClipTests(unittest.TestCase):
    def test_outbox_keeps_clip_next_to_record(self):
        from detector.api_client import ApiClient
        with tempfile.TemporaryDirectory() as d:
            c = ApiClient("http://127.0.0.1:9", "k", outbox_dir=os.path.join(d, "out"))
            c.send_violation({"type": "speeding", "detected_at": "2026-11-10T10:00:00"}, b"\xff\xd8\xff", video=b"....ftypmp42")
            names = sorted(n for n in os.listdir(os.path.join(d, "out")) if "." in n)
            self.assertEqual([n.rsplit(".", 1)[1] for n in names], ["jpg", "json", "mp4"])


if __name__ == "__main__":
    unittest.main()


class FocusedPhotoTests(unittest.TestCase):
    def test_photo_is_drawn_for_the_violating_vehicle_only(self):
        import types
        import cv2
        from detector.pipeline import Pipeline
        pl = Pipeline(None, None, None)
        frame = np.zeros((720, 1280, 3), np.uint8)
        a = types.SimpleNamespace(track_id=7, box=(100, 200, 300, 400))
        b = types.SimpleNamespace(track_id=8, box=(500, 200, 700, 400))
        seen = {}

        def drawer(image, tracks, focus):
            seen["focus"], seen["n"] = focus, len(tracks)
            out = image.copy()
            out[:, :, 1] = 255
            return out
        pl.photo_drawer = drawer
        pl._best[7] = (100.0, frame, a.box, [a, b])
        reading = types.SimpleNamespace(track_id=7, speed_kmh=25.0)
        jpg = pl._evidence(reading, "plate ABC1234")
        self.assertEqual((seen["focus"], seen["n"]), (7, 2))
        self.assertGreater(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)[300:, :, 1].mean(), 200)
        # no drawer: the plain photo with a red box still works
        pl.photo_drawer = None
        self.assertTrue(pl._evidence(reading))
        # the frame where the plate box was found is preferred, so the box fits the picture
        pl.photo_drawer = drawer
        other = np.full((720, 1280, 3), 5, np.uint8)
        pl._plate_frame[7] = (other, a.box, [a])
        pl._evidence(reading)
        self.assertEqual(seen["n"], 1)
        pl._release(7)
        self.assertNotIn(7, pl._plate_frame)


class ParkedVehicleTests(unittest.TestCase):
    def test_parked_is_ignored_even_when_track_ids_keep_changing(self):
        import types
        from detector.pipeline import Pipeline
        pl = Pipeline(None, None, None, parked_after_s=15.0)
        box = lambda tid, x: types.SimpleNamespace(track_id=tid, box=(x, 100, x + 200, 250), name="car")
        parked, mover = [], []
        for sec in range(0, 30):
            parked.append(pl._is_parked(box(1000 + sec, 100 + (sec % 3)), float(sec)))   # new id every second, jitters
            mover.append(pl._is_parked(box(2, 400 + sec * 40), float(sec)))
        self.assertFalse(parked[5])
        self.assertTrue(parked[20])
        self.assertFalse(any(mover))
        # the parked car drives off: far from its place, so it counts again
        self.assertFalse(pl._is_parked(box(5000, 900), 31.0))


class VehicleClassMappingTests(unittest.TestCase):
    def test_roboflow_class_names_map_to_roadsense_classes(self):
        from data_prep.prepare_vehicles import map_class
        cases = {"Jeepney": "jeepney", "jeep": "jeepney", "Tricycles": "tricycle", "tuk-tuk": "tricycle",
                 "Van": "van", "Motorcycles": "motorcycle", "Pickup-Truck": "car", "Toyota Hilux": "car", "Multicab": "multicab", "Multi-Cab": "multicab", "Trailer-Truck": "truck",
                 "Flatbed-Truck": "truck", "UV Express": "van", "Modern Jeepney": "jeepney", "Private Vehicles": "car",
                 "Taxi": "car", "SUV": "car", "Light Truck": "truck", "Minibus": "van", "Bus": "bus", "Car": "car", "Bicycle": None, "E-Bike": None,
                 "person": None}
        for name, expected in cases.items():
            self.assertEqual(map_class(name), expected, name)
