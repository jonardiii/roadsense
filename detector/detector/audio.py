"""Horn detection from a microphone.

A vehicle horn is a loud, steady, tonal sound: a few strong frequencies (usually 300 to 600 Hz plus
harmonics) that hold still for a quarter of a second or more. Speech, engines, wind and tyre noise
are not like that. So a frame of sound counts as "horn-like" only if all of these hold:

  1. It is clearly louder than the recent background (a rolling baseline, so it adapts to a quiet
     morning or a noisy afternoon).
  2. Most of its energy sits in a few narrow spectral peaks (tonal), not spread out like noise.
  3. Its spectrum looks like the previous horn-like frame (steady). Speech and music keep changing.

A burst of such frames lasting `min_duration_s` is one horn event. Beeps within `cooldown_s` of each
other are one event, so a driver tapping the horn three times is reported once.

The thresholds are starting values. Tune them with your own microphone using horn_tool.py.
"""
import collections
import dataclasses
import logging
import os
import queue
import threading
import time
import wave
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("roadsense.audio")

EPS = 1e-12


@dataclass
class HornEvent:
    start: float          # wall-clock time (seconds since 1970) of the first horn-like frame
    end: float            # wall-clock time when the horn was confirmed
    duration_s: float     # how long it had lasted when confirmed
    peak_hz: float        # strongest frequency
    level_db: float       # loudest frame, dB relative to full scale (negative numbers)
    excess_db: float      # how far above the background that was
    tonal: float          # 0 to 1, how concentrated in narrow peaks (1 = a pure tone)
    classifier_score: float = None   # YAMNet's horn score, when the horn model is used
    sound_note: str = None           # what the sound model heard, for the administrator, e.g. 'Vehicle horn 0.82; Engine 0.30'


@dataclass
class FrameInfo:
    """What the detector saw in the latest frame. Used by the live meter."""
    t: float
    level_db: float
    baseline_db: float
    excess_db: float
    tonal: float
    peak_hz: float
    similarity: float
    hot: bool


class HornDetector:
    def __init__(self, sample_rate=16000, frame_s=0.064, hop_s=0.032, band=(250.0, 3500.0),
                 min_excess_db=10.0, min_tonal=0.6, min_similarity=0.85,
                 min_duration_s=0.2, max_gap_s=0.10, cooldown_s=2.0, baseline_s=20.0,
                 peak_range=(200.0, 2500.0), max_run_s=6.0, warmup_s=1.0):
        self.sr = int(sample_rate)
        self.n = int(round(frame_s * self.sr))
        self.hop = int(round(hop_s * self.sr))
        self.window = np.hanning(self.n).astype(np.float32)
        freqs = np.fft.rfftfreq(self.n, 1.0 / self.sr)
        self.band_idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
        self.band_freqs = freqs[self.band_idx]
        self.peak_range = peak_range
        self.min_excess_db = min_excess_db
        self.min_tonal = min_tonal
        self.min_similarity = min_similarity
        self.min_duration_s = min_duration_s
        self.max_gap_s = max_gap_s
        self.cooldown_s = cooldown_s
        self.max_run_s = max_run_s
        self.warmup_frames = int(warmup_s / hop_s)
        self.levels = collections.deque(maxlen=max(10, int(baseline_s / hop_s)))
        self.reset()

    def reset(self):
        self._buf = np.zeros(0, np.float32)
        self._buf_t0 = None
        self._prev_spec = None       # spectrum of the previous frame
        self._run = None             # the horn-like burst being followed
        self._cooldown_until = -1e18
        self.last = None


    def feed(self, samples, t_start):
        """Add mono samples in -1..1 whose first sample was recorded at wall time `t_start`.
        Returns the horn events confirmed within them."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self._buf_t0 is None or len(self._buf) == 0:
            self._buf_t0 = t_start
        else:
            expected = self._buf_t0 + len(self._buf) / self.sr
            gap = t_start - expected
            if abs(gap) > 1.0:           # the audio stream was interrupted: start over
                self.reset()
                self._buf_t0 = t_start
            elif abs(gap) > 0.1:         # sound card clock drifting against the wall clock
                self._buf_t0 += gap
        self._buf = np.concatenate([self._buf, samples])
        events = []
        pos = 0
        while pos + self.n <= len(self._buf):
            frame = self._buf[pos:pos + self.n]
            t = self._buf_t0 + pos / self.sr
            ev = self._process_frame(frame, t)
            if ev:
                events.append(ev)
            pos += self.hop
        self._buf = self._buf[pos:]
        self._buf_t0 += pos / self.sr
        return events


    def _process_frame(self, frame, t):
        level_db = 10.0 * np.log10(float(np.mean(frame.astype(np.float64) ** 2)) + EPS)
        spec = np.abs(np.fft.rfft(frame * self.window))[self.band_idx].astype(np.float64)
        power = spec ** 2
        total = float(power.sum()) + EPS
        tonal, peak_hz = self._tonality(power, total)

        baseline = float(np.median(self.levels)) if len(self.levels) >= self.warmup_frames else None
        excess = level_db - baseline if baseline is not None else 0.0

        sim = 1.0
        if self._prev_spec is not None:
            sim = self._cosine(spec, self._prev_spec)
        self._prev_spec = spec

        hot = (baseline is not None and excess >= self.min_excess_db and tonal >= self.min_tonal
               and self.peak_range[0] <= peak_hz <= self.peak_range[1])
        self.last = FrameInfo(t, level_db, baseline if baseline is not None else level_db,
                              excess, tonal, peak_hz, sim, hot)

        event = None
        in_run = self._run is not None and t - self._run["last_hot"] <= self.max_gap_s + 1e-9
        if hot:
            if not in_run:
                self._run = {"start": t, "last_hot": t, "emitted": False, "specs": [], "peak": [],
                             "level": level_db, "excess": excess, "tonal": []}
            run = self._run
            run["last_hot"] = t
            run["specs"].append(power)
            run["peak"].append(peak_hz)
            run["level"] = max(run["level"], level_db)
            run["excess"] = max(run["excess"], excess)
            run["tonal"].append(tonal)
            length = t - run["start"] + self.n / self.sr
            if not run["emitted"] and length >= self.min_duration_s and self._steady(run):
                run["emitted"] = True       # a burst inside the cooldown is part of the last event
                if t >= self._cooldown_until:
                    event = HornEvent(start=run["start"], end=t + self.n / self.sr, duration_s=length,
                                      peak_hz=float(np.median(run["peak"])), level_db=float(run["level"]),
                                      excess_db=float(run["excess"]), tonal=float(np.mean(run["tonal"])))
            if run["emitted"]:
                self._cooldown_until = max(self._cooldown_until, t + self.cooldown_s)
        elif self._run is not None and t - self._run["last_hot"] > self.max_gap_s:
            self._run = None

        # The background baseline learns from everything that is not a horn-like frame. A tone that
        # lasts longer than max_run_s (a siren, a generator) is treated as background from then on.
        long_run = self._run is not None and t - self._run["start"] > self.max_run_s
        if not hot or long_run:
            self.levels.append(level_db)
        return event

    def _steady(self, run):
        """A horn holds the same sound. Compare each frame of the burst with the burst's typical
        spectrum. The first frame is skipped because it is only partly filled with sound. Speech, a
        rising engine and a siren sweep all drift away from it."""
        specs = np.array(run["specs"][1:][-40:])
        if len(specs) < 3:
            return False
        ref = np.median(specs, axis=0)
        close = [self._cosine(sp, ref) >= self.min_similarity for sp in specs]
        return sum(close) / len(close) >= 0.8

    @staticmethod
    def _cosine(a, b):
        return float(np.dot(a, b) / ((np.linalg.norm(a) + EPS) * (np.linalg.norm(b) + EPS)))

    def _tonality(self, power, total):
        """Share of the band's energy held by its six strongest narrow peaks, and the strongest
        peak's frequency."""
        centre = power[1:-1]
        is_peak = (centre > power[:-2]) & (centre >= power[2:])
        idx = np.where(is_peak)[0] + 1
        if len(idx) == 0:
            return 0.0, 0.0
        top = idx[np.argsort(power[idx])[-6:]]
        held = np.zeros(len(power), bool)
        for i in top:
            held[max(0, i - 2):i + 3] = True
        return float(power[held].sum() / total), float(self.band_freqs[top[-1]])



class AudioListener:
    """Reads the microphone in the background and collects horn events. Call poll() from the main
    loop to get new ones."""

    def __init__(self, detector, device=None, on_chunk=None):
        self.sd = None
        self.detector = detector
        self.device = device
        self.on_chunk = on_chunk
        self.classifier = None   # HornClassifier: a second opinion that filters out shouting and other sounds
        self.min_score = 0.3
        self._ring = collections.deque()    # (wall time of first sample, samples) of the last seconds
        self._pending = []                  # events waiting for the sound that follows them
        self.tap = None          # optional function(samples, wall time of first sample), e.g. the clip recorder
        self.events = queue.Queue()
        self._chunks = queue.Queue(maxsize=400)
        self._stop = threading.Event()
        self._stream = None
        self._thread = None
        self.dropped = 0

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.debug("Audio status: %s", status)
        latency = time_info.currentTime - time_info.inputBufferAdcTime
        if not 0.0 <= latency < 1.0:           # some Windows drivers report nonsense
            latency = frames / self.detector.sr
        try:
            self._chunks.put_nowait((indata[:, 0].copy(), time.time() - latency))
        except queue.Full:
            self.dropped += 1

    def _worker(self):
        while not self._stop.is_set():
            try:
                samples, t0 = self._chunks.get(timeout=0.2)
            except queue.Empty:
                continue
            self._handle(samples, t0)

    def _handle(self, samples, t0):
        if self.tap:
            try:
                self.tap(samples, t0)
            except Exception:  # noqa: BLE001 - recording sound must never stop horn detection
                log.debug("Audio tap failed", exc_info=True)
        if self.classifier:
            self._ring.append((t0, samples))
            while self._ring and t0 - self._ring[0][0] > 12.0:
                self._ring.popleft()
        for ev in self.detector.feed(samples, t0):
            if self.classifier:
                self._pending.append(ev)       # judged a moment later, when its sound has been recorded
            else:
                self.events.put(ev)
        if self._pending:
            self._judge_pending(t0 + len(samples) / self.detector.sr)
        if self.on_chunk:
            self.on_chunk(self.detector.last)

    def _sound(self, start, end):
        """Recorded samples between two wall times."""
        sr = self.detector.sr
        parts = []
        for t0, s in self._ring:
            if t0 + len(s) / sr < start or t0 > end:
                continue
            a, b = max(0, int((start - t0) * sr)), min(len(s), int((end - t0) * sr))
            if b > a:
                parts.append(s[a:b])
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def _judge_pending(self, now):
        keep = []
        for ev in self._pending:
            if now < ev.end + 0.8:
                keep.append(ev)
                continue
            sound = self._sound(ev.start - 0.3, ev.end + 0.8)
            try:
                verdict = self.classifier.judge(sound, self.detector.sr)
            except Exception:  # noqa: BLE001 - if the model fails, keep the event: the frequency check passed
                log.warning("Horn classifier failed; keeping the event.", exc_info=True)
                self.events.put(ev)
                continue
            if verdict.accepted(self.min_score):
                log.info("Horn confirmed by the sound model: %s", verdict.note())
                self.events.put(dataclasses.replace(ev, classifier_score=verdict.horn, sound_note=verdict.note()))
            else:
                log.info("Not a horn. The sound model heard: %s (horn %.2f, voice %.2f, alarm %.2f; the horn needs "
                         "%.2f and must be the strongest). Ignored.", verdict.note(), verdict.horn, verdict.voice,
                         verdict.alarm, self.min_score)
        self._pending = keep

    def start(self):
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("Horn detection needs the sounddevice package: pip install sounddevice") from exc
        self.sd = sd
        block = int(self.detector.sr * 0.032)
        self._stream = self.sd.InputStream(samplerate=self.detector.sr, channels=1, dtype="float32",
                                           blocksize=block, device=self.device, callback=self._callback)
        self._stream.start()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        log.info("Listening for horns on '%s' at %d Hz.", self._device_name(), self.detector.sr)

    def _device_name(self):
        try:
            return self.sd.query_devices(self.device, "input")["name"]
        except Exception:  # noqa: BLE001
            return str(self.device)

    def poll(self):
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def stop(self):
        self._stop.set()
        if self._stream:
            self._stream.stop()
            self._stream.close()
        if self._thread:
            self._thread.join(timeout=1)


class FileAudioListener(AudioListener):
    """The sound of a video file, fed to the horn detector in step with the picture.

    For testing with a recorded or downloaded video (the microphone cannot hear a recording). Call
    advance(frame) for every frame: the sound up to that frame's time is processed, so a horn is judged
    together with the vehicles in the very picture where it happened. The camera of a live system never
    uses this: its sound always comes from the external microphone.
    """

    def __init__(self, detector, samples):
        super().__init__(detector)
        self.samples = samples
        self._base = None        # wall time of the first sample = wall time of the video's time 0
        self._pos = 0

    def start(self):
        log.info("Horn detection uses the sound of the video file (%.0f s).", len(self.samples) / self.detector.sr)

    def advance(self, frame):
        if self._base is None:
            self._base = frame.wall.timestamp() - frame.t
        self._feed_until(frame.wall.timestamp() - self._base)

    def _feed_until(self, seconds):
        sr = self.detector.sr
        step = int(sr * 0.032)
        end = min(len(self.samples), int(seconds * sr))
        while self._pos + step <= end:
            self._handle(self.samples[self._pos:self._pos + step], self._base + self._pos / sr)
            self._pos += step

    def stop(self):
        """The video ended: process the rest of the sound, then a second of quiet so the last event is judged."""
        if self._base is None:
            return
        sr = self.detector.sr
        self._feed_until(len(self.samples) / sr)
        quiet = np.zeros(int(sr * 0.032), np.float32)
        for i in range(int(1.2 / 0.032)):
            self._handle(quiet, self._base + (len(self.samples) + i * len(quiet)) / sr)


def decode_file_audio(path, sample_rate=16000, exe=None):
    """The sound track of a video file as mono float samples, or None when it has none (or no ffmpeg)."""
    import subprocess

    from .clips import ffmpeg_path
    exe = exe or ffmpeg_path()
    if not exe:
        return None
    cmd = [exe, "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"]
    flags = 0x08000000 if os.name == "nt" else 0
    try:
        out = subprocess.run(cmd, capture_output=True, creationflags=flags, timeout=600).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if len(out) < sample_rate * 2:      # under a second: no sound track
        return None
    return np.frombuffer(out[:len(out) // 2 * 2], "<i2").astype(np.float32) / 32768.0



def load_wav(path):
    """Read a WAV file as mono float samples in -1..1. Returns (samples, sample_rate)."""
    try:
        with wave.open(path, "rb") as w:
            sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(n)
        if width == 2:
            data = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
        elif width == 1:
            data = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 4:
            data = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
        elif width == 3:
            b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
            v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            v = np.where(v >= 1 << 23, v - (1 << 24), v)
            data = v.astype(np.float32) / (1 << 23)
        else:
            raise ValueError(f"Unsupported WAV sample width: {width} bytes")
    except wave.Error:
        from scipy.io import wavfile      # float WAV files
        sr, data = wavfile.read(path)
        ch = data.shape[1] if data.ndim > 1 else 1
        data = data.astype(np.float32).reshape(-1)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data.astype(np.float32), sr


def save_wav(path, samples, sample_rate):
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def detect_in_samples(samples, sample_rate, chunk_s=0.1, **kw):
    """Run the detector over recorded sound. Returns its horn events (times start at 0)."""
    det = HornDetector(sample_rate, **kw)
    step = int(chunk_s * sample_rate)
    events = []
    for i in range(0, len(samples), step):
        events.extend(det.feed(samples[i:i + step], i / sample_rate))
    return events
