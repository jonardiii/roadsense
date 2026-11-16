"""Short video clips as evidence: a few seconds before a violation and a few seconds after it.

The newest frames are kept in memory as small JPEGs. When a violation happens the clip is cut from
them, the frames that arrive afterwards are added, and the whole thing is encoded to an MP4 that a
browser can play. Encoding runs in a background thread so the detector never stops for it.

It needs ffmpeg. `pip install imageio-ffmpeg` brings a copy, nothing else to install.
"""
import collections
import io
import logging
import os
import subprocess
import tempfile
import threading
import wave
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

log = logging.getLogger("roadsense.clips")
MAX_CLIP_BYTES = 25 * 1024 * 1024   # the website accepts up to 30 MB


def ffmpeg_path():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - missing package or no bundled binary
        return None


class ClipRecorder:
    def __init__(self, before_s=7.0, after_s=3.0, width=960, fps=12.0, crf=28, quality=70):
        self.audio_rate = 16000
        self.audio_lag_s = 0.0      # the picture arrives this much later than the sound (set by the program)
        self.audio_on = False       # set True when a microphone feeds add_audio()
        self._audio = collections.deque()   # (wall time of first sample, int16 samples)
        self.before_s, self.after_s = float(before_s), float(after_s)
        self.width, self.fps, self.crf, self.quality = int(width), float(fps), int(crf), int(quality)
        self.exe = ffmpeg_path()
        self._keep_s = self.before_s + self.after_s + 6.0
        self._frames = collections.deque()      # (time, jpeg bytes), newest last
        self._last_added = -1e9
        self._jobs = []                          # clips still waiting for their last frames
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clips")

    @property
    def ready(self):
        return self.exe is not None

    def add(self, image, stamp):
        """Offer every frame. Frames closer together than 1/fps are skipped."""
        if not self.ready:                  # no ffmpeg: no video, but sound clips still need the clock
            self._check_jobs(stamp)
            return
        if stamp - self._last_added < 1.0 / self.fps - 1e-6:
            self._check_jobs(stamp)
            return
        self._last_added = stamp
        h, w = image.shape[:2]
        new_w = min(self.width, w) // 2 * 2                  # encoders need even sizes
        new_h = max(2, int(round(h * new_w / w / 2)) * 2)
        small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        item = (stamp, buf.tobytes())
        with self._lock:
            self._frames.append(item)
            while self._frames and stamp - self._frames[0][0] > self._keep_s:
                self._frames.popleft()
            for job in self._jobs:
                if stamp <= job["end"]:
                    job["frames"].append(item)
        self._check_jobs(stamp)

    def add_audio(self, samples, t0):
        """Microphone samples (-1..1) whose first sample was recorded at wall time t0."""
        pcm = (np.clip(np.asarray(samples, np.float32).reshape(-1), -1.0, 1.0) * 32767.0).astype(np.int16)
        with self._lock:
            self._audio.append((t0, pcm))
            horizon = t0 - (self._keep_s + self.audio_lag_s + 6.0)
            while self._audio and self._audio[0][0] < horizon:
                self._audio.popleft()

    def _audio_for(self, start, end):
        """Sound for the picture window [start, end]. The picture is late by audio_lag_s, so the sound
        for it was recorded that much earlier. Returns int16 bytes exactly as long as the window, or None."""
        if not self.audio_on:
            return None
        sr = self.audio_rate
        n = int(round((end - start) * sr))
        t_from = start - self.audio_lag_s
        out = np.zeros(n, np.int16)
        filled = 0
        with self._lock:
            chunks = list(self._audio)
        for t0, pcm in chunks:
            i = int(round((t0 - t_from) * sr))
            lo, hi = max(0, i), min(n, i + len(pcm))
            if hi > lo:
                out[lo:hi] = pcm[lo - i:hi - i]
                filled += hi - lo
        return out.tobytes() if n > 0 and filled >= 0.5 * n else None

    def request(self, event_stamp, callback, with_audio=False):
        """Cut a clip around event_stamp. callback(mp4 bytes or None) is called when it is ready; with
        with_audio=True it is callback(mp4 or None, wav or None): the wav is the microphone's own sound for
        the same seconds, as a separate file."""
        if not self.ready and not (with_audio and self.audio_on):
            callback(None, None) if with_audio else callback(None)
            return
        start, end = event_stamp - self.before_s, event_stamp + self.after_s
        with self._lock:
            frames = [f for f in self._frames if start <= f[0] <= end]
            self._jobs.append({"start": start, "end": end, "frames": frames, "callback": callback,
                               "with_audio": with_audio})
        if self._frames:
            self._check_jobs(self._frames[-1][0])

    def _check_jobs(self, now):
        done = []
        with self._lock:
            for job in list(self._jobs):
                if now >= job["end"]:
                    self._jobs.remove(job)
                    done.append(job)
        for job in done:
            self._pool.submit(self._finish, job)

    def flush(self):
        """Finish every waiting clip with the frames it has (used when the program stops)."""
        with self._lock:
            jobs, self._jobs = self._jobs, []
        for job in jobs:
            self._pool.submit(self._finish, job)

    def close(self):
        self.flush()
        self._pool.shutdown(wait=True)

    def _wav(self, start, end):
        """The microphone's sound for [start, end] as a WAV file, or None when it was not recorded."""
        sound = self._audio_for(start, end)
        if not sound:
            return None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.audio_rate)
            w.writeframes(sound)
        return buf.getvalue()

    def _finish(self, job):
        video = audio = None
        if self.ready:
            try:
                video = self._encode(job["frames"], job["start"], job["end"])
            except Exception as exc:  # noqa: BLE001 - the violation must still be sent without a clip
                log.warning("Could not make the video clip (%s). Sending the violation with the photo only.", exc)
        if job.get("with_audio"):
            try:
                audio = self._wav(job["start"], job["end"])
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not make the sound clip (%s).", exc)
        try:
            job["callback"](video, audio) if job.get("with_audio") else job["callback"](video)
        except Exception:  # noqa: BLE001
            log.exception("Delivering a violation with its clip failed")

    def _encode(self, frames, start, end):
        if len(frames) < 2:
            return None
        first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
        h, w = first.shape[:2]
        stop = min(end, frames[-1][0])
        start = max(start, frames[0][0])
        count = int((stop - start) * self.fps) + 1
        fd, path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        apath = None
        try:
            seconds = count / self.fps
            sound = self._audio_for(start, start + seconds)
            cmd = [self.exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{w}x{h}", "-r", f"{self.fps:g}", "-i", "-"]
            if sound:
                afd, apath = tempfile.mkstemp(suffix=".raw")
                with os.fdopen(afd, "wb") as fh:
                    fh.write(sound)
                cmd += ["-f", "s16le", "-ar", str(self.audio_rate), "-ac", "1", "-i", apath,
                        "-c:a", "aac", "-b:a", "64k", "-shortest"]
            else:
                cmd += ["-an"]
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(self.crf), "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", path]
            flags = 0x08000000 if os.name == "nt" else 0     # no console window on Windows
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
            j, current = 0, None
            for k in range(count):
                tk = start + k / self.fps
                while j < len(frames) and frames[j][0] <= tk:
                    j += 1
                    current = None
                if current is None:
                    jpeg = frames[max(0, j - 1)][1]
                    current = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                proc.stdin.write(current.tobytes())
            proc.stdin.close()
            err = proc.stderr.read().decode(errors="replace")
            if proc.wait(timeout=120) != 0:
                raise RuntimeError(err.strip()[-200:] or "ffmpeg failed")
            with open(path, "rb") as fh:
                data = fh.read()
            return data if 0 < len(data) <= MAX_CLIP_BYTES else None
        finally:
            for leftover in (path, apath):
                try:
                    if leftover:
                        os.remove(leftover)
                except OSError:
                    pass
