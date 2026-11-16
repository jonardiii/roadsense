import datetime
import io
import os
import tempfile
import types
import unittest
import wave

import numpy as np

from detector.audio import FileAudioListener, HornDetector
from detector.clips import ClipRecorder

SR = 16000


def horn(seconds, hz=450.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * hz * t) + 0.25 * np.sin(2 * np.pi * 2 * hz * t)).astype(np.float32)


class SoundClipTests(unittest.TestCase):
    def mic(self, rec, seconds, start=1000.0):
        for k in range(int(seconds * 10)):
            rec.add_audio(horn(0.1), start + k * 0.1)

    def test_sound_clip_is_a_separate_wav_of_the_same_seconds(self):
        rec = ClipRecorder(before_s=3, after_s=2)
        rec.audio_on = True
        got = []
        self.mic(rec, 6)
        rec.request(1003.0, lambda video, audio: got.append((video, audio)), with_audio=True)
        for k in range(10):                      # frames carry the clock past the end of the clip
            rec.add(np.zeros((100, 160, 3), np.uint8), 1005.0 + k * 0.2)
        rec.close()
        self.assertEqual(len(got), 1)
        wav = got[0][1]
        self.assertTrue(wav and wav[:4] == b"RIFF")
        with wave.open(io.BytesIO(wav)) as w:
            self.assertEqual(w.getframerate(), SR)
            self.assertAlmostEqual(w.getnframes() / SR, 5.0, delta=0.1)

    def test_sound_clip_works_without_ffmpeg_and_none_without_microphone(self):
        rec = ClipRecorder(before_s=3, after_s=2)
        rec.exe = None
        rec.audio_on = True
        got = []
        self.mic(rec, 6)
        rec.request(1003.0, lambda video, audio: got.append((video, audio)), with_audio=True)
        rec.add(np.zeros((10, 10, 3), np.uint8), 1006.0)
        rec.close()
        self.assertIsNone(got[0][0])
        self.assertTrue(got[0][1])
        quiet = ClipRecorder()
        quiet.exe = None
        out = []
        quiet.request(5.0, lambda video, audio: out.append((video, audio)), with_audio=True)
        self.assertEqual(out, [(None, None)])

    def test_outbox_keeps_sound_clip_and_sends_it(self):
        from detector.api_client import ApiClient
        sent = []

        class Client(ApiClient):
            def _post_violation(self, fields, jpeg, video=None, audio=None):
                sent.append((jpeg, video, audio))
                return types.SimpleNamespace(status_code=201, text="")
        with tempfile.TemporaryDirectory() as d:
            c = Client("http://127.0.0.1:9", "k", outbox_dir=os.path.join(d, "out"))
            c._enqueue({"type": "honking", "detected_at": "2026-11-10T10:00:00"}, b"\xff\xd8\xff", None, b"RIFF....WAVE")
            names = sorted(n.rsplit(".", 1)[1] for n in os.listdir(os.path.join(d, "out")) if "." in n)
            self.assertEqual(names, ["jpg", "json", "wav"])
            self.assertEqual(c.flush_outbox(), 1)
            self.assertEqual(sent[0][2], b"RIFF....WAVE")
            self.assertEqual([n for n in os.listdir(os.path.join(d, "out")) if "." in n], [])


class FileAudioTests(unittest.TestCase):
    def test_sound_of_a_video_file_is_fed_in_step_with_the_frames(self):
        quiet = np.random.default_rng(1).standard_normal(3 * SR).astype(np.float32) * 0.002
        samples = np.concatenate([quiet, horn(1.0), np.zeros(2 * SR, np.float32)])   # horn at 3.0 s .. 4.0 s
        det = HornDetector(SR, min_duration_s=0.2, cooldown_s=2.0)
        listener = FileAudioListener(det, samples)
        listener.start()
        start = datetime.datetime(2026, 11, 10, 10, 0, 0, tzinfo=datetime.timezone.utc)
        found = []
        for i in range(0, 6 * 10):               # 10 frames a second, 6 seconds
            t = i / 10.0
            listener.advance(types.SimpleNamespace(t=t, wall=start + datetime.timedelta(seconds=t)))
            found += listener.poll()
        listener.stop()
        found += listener.poll()
        self.assertEqual(len(found), 1)
        # the event is stamped on the same clock as the frames: 3.0 s after the start
        self.assertAlmostEqual(found[0].start - start.timestamp(), 3.0, delta=0.3)

    def test_nothing_is_heard_before_its_frame_arrives(self):
        samples = np.concatenate([np.zeros(SR, np.float32), horn(1.0)])
        listener = FileAudioListener(HornDetector(SR), samples)
        start = datetime.datetime(2026, 11, 10, 10, 0, 0, tzinfo=datetime.timezone.utc)
        listener.advance(types.SimpleNamespace(t=0.0, wall=start))
        self.assertEqual(listener._pos, 0)
        listener.advance(types.SimpleNamespace(t=1.0, wall=start + datetime.timedelta(seconds=1)))
        self.assertAlmostEqual(listener._pos / SR, 1.0, delta=0.05)


class CameraAudioTests(unittest.TestCase):
    def setUp(self):
        self.saved = os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)

    def tearDown(self):
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        if self.saved is not None:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = self.saved

    def test_rtsp_camera_is_asked_for_video_only(self):
        from detector.video import FrameSource
        FrameSource("rtsp://u:p@192.168.1.5:554/stream1")._no_camera_audio()
        opts = os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
        self.assertIn("allowed_media_types;video", opts)
        self.assertIn("rtsp_transport;tcp", opts)      # OpenCV's own default is kept

    def test_user_options_are_kept_and_other_sources_untouched(self):
        from detector.video import FrameSource
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "stimeout;5000000"
        FrameSource("rtsp://cam/stream")._no_camera_audio()
        self.assertEqual(os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"], "stimeout;5000000|allowed_media_types;video")
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        FrameSource("https://example.com/live.m3u8")._no_camera_audio()
        FrameSource("rtsp://cam/stream", camera_audio_off=False)._no_camera_audio()
        self.assertNotIn("OPENCV_FFMPEG_CAPTURE_OPTIONS", os.environ)


if __name__ == "__main__":
    unittest.main()
