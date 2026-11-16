"""Test and tune horn detection with your own microphone. No camera or web system needed.

  python horn_tool.py devices                      list microphones
  python horn_tool.py meter                        live meter; honk, talk, clap and watch it
  python horn_tool.py record --seconds 10 --out horn1.wav   save a sound for testing
  python horn_tool.py analyze horn1.wav            list the horns found in a recording
  python horn_tool.py evaluate --horn data/horn --other data/other
                                                   count hits and false alarms over two folders of WAV files
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

from detector.audio import AudioListener, HornDetector, detect_in_samples, load_wav, save_wav


def tuning_args(ap):
    ap.add_argument("--min-excess-db", type=float, default=10.0, help="how far above the background (default 10)")
    ap.add_argument("--min-tonal", type=float, default=0.6, help="0 to 1, tone-likeness (default 0.6)")
    ap.add_argument("--min-similarity", type=float, default=0.85, help="steadiness 0 to 1 (default 0.85)")
    ap.add_argument("--min-duration", type=float, default=0.2, help="seconds (default 0.2)")
    ap.add_argument("--cooldown", type=float, default=2.0, help="seconds (default 2)")


def kwargs(a):
    return dict(min_excess_db=a.min_excess_db, min_tonal=a.min_tonal, min_similarity=a.min_similarity,
                min_duration_s=a.min_duration, cooldown_s=a.cooldown)


def device_arg(value):
    if value is None or value == "":
        return None
    return int(value) if str(value).isdigit() else value


def cmd_devices(_):
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"{i:3d}  {d['name']}  ({d['max_input_channels']} ch, default {d['default_samplerate']:.0f} Hz)")
    print("\nUse the number in the config as audio_device, or leave it empty for the default.")


def cmd_meter(a):
    det = HornDetector(a.rate, **kwargs(a))
    listener = AudioListener(det, device=device_arg(a.device))
    listener.start()
    print("Listening. Honk, talk, clap, drive past. Ctrl+C to stop.\n"
          "level and baseline are dB (0 = loudest possible); excess = level - baseline.\n"
          "A horn needs excess >= %.0f, tone >= %.2f, and to hold steady for %.2f s.\n"
          % (a.min_excess_db, a.min_tonal, a.min_duration))
    try:
        while True:
            time.sleep(0.1)
            f = det.last
            if f:
                bar = "#" * max(0, min(40, int((f.level_db + 70) / 70 * 40)))
                print(f"\r{f.level_db:6.1f} dB  base {f.baseline_db:6.1f}  excess {f.excess_db:5.1f}  "
                      f"tone {f.tonal:4.2f}  {f.peak_hz:5.0f} Hz  {'HORN-LIKE' if f.hot else '          '} {bar:<40}",
                      end="", flush=True)
            for ev in listener.poll():
                print(f"\n>>> HORN: {ev.duration_s:.2f} s at {ev.peak_hz:.0f} Hz, {ev.excess_db:.0f} dB above "
                      f"background (level {ev.level_db:.1f} dB, tone {ev.tonal:.2f})")
    except KeyboardInterrupt:
        print()
    finally:
        listener.stop()


def cmd_record(a):
    import sounddevice as sd
    print(f"Recording {a.seconds} s from the microphone...")
    x = sd.rec(int(a.seconds * a.rate), samplerate=a.rate, channels=1, dtype="float32",
               device=device_arg(a.device))
    sd.wait()
    save_wav(a.out, x[:, 0], a.rate)
    peak = float(np.abs(x).max())
    print(f"Saved {a.out}. Loudest sample: {peak:.2f}" + ("  (too loud: it is clipping, lower the mic gain)" if peak > 0.98 else ""))


def cmd_analyze(a):
    for path in a.files:
        samples, sr = load_wav(path)
        events = detect_in_samples(samples, sr, **kwargs(a))
        print(f"{path}: {len(samples) / sr:.1f} s, {len(events)} horn(s)")
        for ev in events:
            print(f"   at {ev.start:6.2f} s  lasted {ev.duration_s:.2f} s  {ev.peak_hz:.0f} Hz  "
                  f"{ev.excess_db:.0f} dB above background  tone {ev.tonal:.2f}")


def cmd_evaluate(a):
    def files(folder):
        return sorted(glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True))

    horn_files, other_files = files(a.horn), files(a.other)
    if not horn_files or not other_files:
        sys.exit("Need .wav files in both folders.")
    missed, alarms, other_seconds = [], [], 0.0
    for path in horn_files:
        samples, sr = load_wav(path)
        if not detect_in_samples(samples, sr, **kwargs(a)):
            missed.append(path)
    for path in other_files:
        samples, sr = load_wav(path)
        other_seconds += len(samples) / sr
        n = len(detect_in_samples(samples, sr, **kwargs(a)))
        if n:
            alarms.append((path, n))
    hit = len(horn_files) - len(missed)
    print(f"Horn recordings: {hit} of {len(horn_files)} detected ({100.0 * hit / len(horn_files):.0f}%).")
    for p in missed:
        print("   missed:", p)
    total_alarms = sum(n for _, n in alarms)
    per_hour = total_alarms / (other_seconds / 3600.0) if other_seconds else 0.0
    print(f"Other sounds: {other_seconds / 60:.1f} minutes, {total_alarms} false alarm(s) ({per_hour:.1f} per hour).")
    for p, n in alarms:
        print(f"   false alarm x{n}: {p}")
    print("\nEach horn recording should contain a horn; the 'other' folder should hold traffic, voices, "
          "engines and anything else the microphone hears on site, with no horn in it.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices").set_defaults(fn=cmd_devices)
    p = sub.add_parser("meter")
    p.add_argument("--device", default="")
    p.add_argument("--rate", type=int, default=16000)
    tuning_args(p)
    p.set_defaults(fn=cmd_meter)
    p = sub.add_parser("record")
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="")
    p.add_argument("--rate", type=int, default=16000)
    p.set_defaults(fn=cmd_record)
    p = sub.add_parser("analyze")
    p.add_argument("files", nargs="+")
    tuning_args(p)
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("evaluate")
    p.add_argument("--horn", required=True, help="folder of recordings that contain a horn")
    p.add_argument("--other", required=True, help="folder of recordings with no horn")
    tuning_args(p)
    p.set_defaults(fn=cmd_evaluate)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
