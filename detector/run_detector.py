"""Run the detector.

   python run_detector.py --config detector_config.json
   python run_detector.py --config detector_config.json --source clip.mp4 --preview
"""
import argparse
import logging
import os
import time

import cv2

from detector import config as config_mod
from detector.api_client import ApiClient
from detector.live import LiveFeed
from detector.pipeline import Pipeline
from detector.speed import Calibration, SpeedEstimator
from detector.tracker import YoloTracker
from detector.video import FrameSource


def draw_preview(image, pipeline, calibration, tracks=None, focus=None):
    """The picture with boxes, speeds and plates. With `focus` (a track number) only that vehicle is
    labelled, in red with its own plate box; the others are thin outlines. Used for the evidence photo."""
    out = calibration.draw(image)
    big = getattr(pipeline.plates, "min_vehicle_width", 120)
    for d in (pipeline.last_tracks if tracks is None else tracks):
        x1, y1, x2, y2 = (int(v) for v in d.box)
        if focus is not None and d.track_id != focus:
            cv2.rectangle(out, (x1, y1), (x2, y2), (150, 150, 150), 1)
            continue
        speed = pipeline.measured.get(d.track_id)
        if speed is None:
            speed = pipeline.estimator.live_speed(d.track_id)
        fast = speed is not None and speed > pipeline.flag_above_kmh
        color = (0, 0, 255) if (fast or focus is not None) else (0, 255, 0)
        thick = 3 if focus is not None else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        shown_plate = pipeline.plates.visible(d.track_id)
        label = f"#{d.track_id} {d.name}"
        if speed is not None:
            label += f"  {speed:.0f} km/h"
        # Far-away vehicles get only a thin box. Writing a label on every one makes a crowd unreadable.
        if focus is not None or (x2 - x1) >= big:
            cv2.putText(out, label, (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        if shown_plate:
            # The plate gets its own box, with the reading written on the plate, not on the vehicle.
            (px1, py1, px2, py2), plate, conf = shown_plate
            cv2.rectangle(out, (px1, py1), (px2, py2), (255, 200, 0), 3)
            text = f"{plate} ({conf:.2f})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            ty = py2 + th + 8 if py2 + th + 14 < out.shape[0] else max(th + 4, py1 - 8)
            cv2.rectangle(out, (px1, ty - th - 6), (px1 + tw + 8, ty + 6), (0, 0, 0), -1)
            cv2.putText(out, text, (px1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2, cv2.LINE_AA)
    return out


def fit_for_screen(image, max_width=1280):
    """Shrink the preview so a 2304-pixel camera picture fits on a laptop screen."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    return cv2.resize(image, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="detector_config.json")
    ap.add_argument("--source", help="override the video source in the config")
    ap.add_argument("--preview", action="store_true", help="show a window (press Q to stop)")
    ap.add_argument("--realtime", action="store_true", help="play a video file at normal speed")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("roadsense")

    cfg = config_mod.load(args.config)
    if args.source:
        cfg.source = args.source
    if not cfg.source:
        raise SystemExit("Set a video source in the config file or pass --source.")
    if not cfg.api_key:
        raise SystemExit("Set api_key in the config file (copy it from the web Settings page).")

    client = ApiClient(cfg.api_url, cfg.api_key, cfg.outbox_dir)
    remote = client.config()
    limit = remote["speed_limit_kmh"] if remote else 10.0
    if not remote:
        log.warning("Could not read the speed limit from the web system. Using %g km/h for now.", limit)

    calibration = Calibration.load(cfg.calibration)
    estimator = SpeedEstimator(calibration, cfg.min_samples, cfg.min_duration_s,
                               cfg.min_distance_m, cfg.max_residual_m)
    tracker = YoloTracker(cfg.model, cfg.device, cfg.imgsz, cfg.conf, cfg.iou,
                          cfg.vehicle_classes, cfg.tracker)
    plate_reader = None
    if cfg.plate_model:
        from detector.plates.ocr import make_ocr
        from detector.plates.reader import PlateDetector, PlateReader
        plate_reader = PlateReader(
            PlateDetector(cfg.plate_model, cfg.device, conf=cfg.plate_conf),
            make_ocr(cfg.ocr_engine, gpu=cfg.ocr_gpu), client.plates() or [], cfg.min_plate_confidence,
            cfg.plate_every_n_frames, cfg.max_plate_reads, min_vehicle_width=cfg.min_vehicle_width_px,
            min_plate_width=cfg.min_plate_width_px, min_sharpness=cfg.min_plate_sharpness,
            max_candidates=cfg.plate_candidates, require_valid_format=cfg.require_valid_format)
        log.info("Plate reading is on (%s, reader: %s).", cfg.plate_model, cfg.ocr_engine)
    else:
        log.warning("Plate reading is off: no plate_model in the config. Violations will be sent "
                    "without a plate.")
    clips = None
    if cfg.clip_enabled:
        from detector.clips import ClipRecorder
        clips = ClipRecorder(cfg.clip_before_s, cfg.clip_after_s, cfg.clip_width, cfg.clip_fps)
        if clips.ready:
            print(f"Video clips on: {cfg.clip_before_s:g} s before and {cfg.clip_after_s:g} s after each violation.")
        else:
            print("Video clips are off: run  pip install imageio-ffmpeg  to turn them on. Photos and the "
                  "separate sound clips still work.")
    # A video file with a sound track (a test with a recording or a downloaded clip) brings its own sound,
    # and the picture and sound are in step. A camera never does: its sound comes from the external microphone.
    file_audio = None
    if cfg.horn_enabled and os.path.isfile(str(cfg.source)):
        from detector.audio import decode_file_audio
        file_audio = decode_file_audio(cfg.source, cfg.audio_rate)
    pipeline = Pipeline(tracker, estimator, client, limit, cfg.speed_margin_kmh,
                        plate_reader=plate_reader, jpeg_quality=cfg.jpeg_quality,
                        video_lag_s=0.0 if file_audio is not None else cfg.video_lag_s, horn_min_box_px=cfg.horn_min_box_px,
                        mic_offset_db=cfg.mic_offset_db, clips=clips,
                        parked_after_s=cfg.parked_after_s,
                        horn_min_level_db=(remote or {}).get("honk_db_threshold"),
                        horn_min_seconds=(remote or {}).get("honk_min_seconds", 0.0),
                        horn_zone=(lambda box, size: calibration.zone.contains(
                            *calibration.to_world((box[0] + box[2]) / 2.0, box[3], size)))
                        if cfg.horn_zone_only else None)
    pipeline.photo_drawer = lambda image, tracks, focus: draw_preview(image, pipeline, calibration, tracks, focus)
    listener = None
    if cfg.horn_enabled:
        from detector.audio import AudioListener, HornDetector
        device = cfg.audio_device.strip()
        listener = AudioListener(
            HornDetector(cfg.audio_rate, min_excess_db=cfg.horn_min_excess_db, min_tonal=cfg.horn_min_tonal,
                         min_similarity=cfg.horn_min_similarity, min_duration_s=cfg.horn_min_duration_s,
                         cooldown_s=cfg.horn_cooldown_s),
            device=(int(device) if device.isdigit() else device or None))
        if os.path.exists(cfg.horn_model):
            try:
                from detector.horn_classifier import HornClassifier
                listener.classifier = HornClassifier(cfg.horn_model, head_path=cfg.horn_head)
                listener.min_score = (cfg.horn_head_min_score or listener.classifier.default_min_score
                                      or cfg.horn_model_min_score)
                if listener.classifier.head is not None:
                    print(f"Horn layer fine-tuned on horn recordings is on ({cfg.horn_head}).")
                print(f"Horn sound model on ({cfg.horn_model}): shouting and other sounds are filtered out.")
            except Exception as exc:  # noqa: BLE001
                print(f"Could not load the horn sound model ({exc}). Using the frequency check only.")
        else:
            print(f"No horn sound model at {cfg.horn_model}: using the frequency check only (see README).")
        if file_audio is not None:
            from detector.audio import FileAudioListener
            file_listener = FileAudioListener(listener.detector, file_audio)
            file_listener.classifier, file_listener.min_score = listener.classifier, listener.min_score
            listener = file_listener
        try:
            listener.start()
        except Exception as exc:  # noqa: BLE001
            log.error("Could not open the microphone (%s). Running without horn detection. "
                      "Run horn_tool.py devices to see the choices.", exc)
            listener = None
    else:
        log.warning("Horn detection is off: set horn_enabled to true in the config to turn it on.")
    source = FrameSource(cfg.source, realtime=args.realtime, camera_audio_off=cfg.camera_audio_off)
    if not source.is_file and cfg.camera_audio_off and str(cfg.source).lower().startswith(("rtsp://", "rtsps://")):
        log.info("The camera's own sound is switched off: only the external microphone is used.")
    if clips and listener:
        if file_audio is not None:
            clips.audio_rate = cfg.audio_rate
            clips.audio_lag_s = 0.0
            clips.audio_on = True
            listener.tap = clips.add_audio
            log.info("Video clips and sound clips use the sound of the video file.")
        elif source.is_file:
            log.info("Video file without sound: clips have no sound (the microphone does not hear a recording).")
        else:
            clips.audio_rate = cfg.audio_rate
            clips.audio_lag_s = cfg.video_lag_s
            clips.audio_on = True
            listener.tap = clips.add_audio
            log.info("Video clips will include the sound from the microphone.")
    elif clips:
        log.info("Video clips have no sound: turn on horn_enabled and plug in the microphone to add it.")
    show = args.preview or cfg.show_preview
    live = None
    if cfg.live_port:
        try:
            live = LiveFeed(cfg.live_port, cfg.live_host)
        except OSError as exc:
            log.warning("Live feed is off (port %s is busy or blocked: %s).", cfg.live_port, exc)

    log.info("Running. Flagging vehicles above %.1f km/h. Press Ctrl+C to stop.", pipeline.flag_above_kmh)
    last_beat = last_cfg = time.monotonic()
    frames, fps_since, fps = 0, time.monotonic(), 0.0
    last = None
    try:
        for frame in source.frames():
            last = frame
            if listener:
                if file_audio is not None:
                    listener.advance(frame)
                for horn in listener.poll():
                    pipeline.add_horn(horn)
            pipeline.process(frame.image, frame.t, frame.wall)
            frames += 1
            now = time.monotonic()
            if now - fps_since >= 5:
                fps, frames, fps_since = frames / (now - fps_since), 0, now
            if now - last_beat >= cfg.heartbeat_s:
                client.heartbeat(f"{fps:.0f} fps, {pipeline.stats['violations']} sent")
                client.flush_outbox()
                last_beat = now
            if now - last_cfg >= cfg.config_refresh_s:
                fresh = client.config()
                if fresh:
                    pipeline.speed_limit_kmh = fresh["speed_limit_kmh"]
                if plate_reader:
                    verified = client.plates()
                    if verified is not None:
                        plate_reader.set_verified(verified)
                last_cfg = now
            # One picture with boxes, speeds and plates serves the live feed, the preview window,
            # the video clip and the photo evidence.
            shown = draw_preview(frame.image, pipeline, calibration)
            if live and live.wanted():
                live.update(shown)
            if clips:
                clips.add(shown, frame.wall.timestamp())
            if show:
                cv2.imshow("RoadSense (Q to stop)", fit_for_screen(shown))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        if live:
            live.close()
        if listener:
            listener.stop()
            for horn in listener.poll():
                pipeline.add_horn(horn)
        if last is not None:
            pipeline.flush(last.wall)   # vehicles still in view when the video ended
        if clips:
            clips.close()               # finish clips still waiting for their last seconds
        sent = client.flush_outbox()
        log.info("Stopped. %d vehicles measured, %d speeding, %d horn violations. %d queued records "
                 "delivered on exit.", pipeline.stats["valid"], pipeline.stats["violations"],
                 pipeline.stats["horns"], sent)
        if client.pending():
            log.warning("%d records are still waiting in '%s'. They send next time the web system is "
                        "reachable.", len(client.pending()), cfg.outbox_dir)
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
