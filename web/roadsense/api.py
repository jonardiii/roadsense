"""JSON API used by the detector program. Every call needs the X-API-Key header."""
import os
import secrets
import uuid

from flask import Blueprint, current_app, jsonify, request

from . import db, sms, utils

bp = Blueprint("api", __name__, url_prefix="/api/v1")


def _fail(message, code):
    return jsonify(error=message), code


@bp.before_request
def _auth():
    key = request.headers.get("X-API-Key", "")
    real = db.get_setting("api_key", "")
    if not key or not real or not secrets.compare_digest(key, real):
        return _fail("Missing or wrong X-API-Key header.", 401)


def _number(value, name, low, high):
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")
    if not low <= num <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return num


@bp.route("/health")
def health():
    return jsonify(ok=True, time=utils.now_str())


@bp.route("/config")
def config():
    """Thresholds the admin sets in the web settings page."""
    return jsonify(
        speed_limit_kmh=float(db.get_setting("speed_limit", "10")),
        honk_db_threshold=float(db.get_setting("honk_db", "85")),
        honk_min_seconds=float(db.get_setting("honk_min_seconds", "1")),
    )


@bp.route("/plates")
def plates():
    """Verified plates, so the detector can match OCR text that is off by a character."""
    rows = db.get_db().execute("SELECT plate FROM vehicles WHERE verified = 1").fetchall()
    return jsonify(plates=[r["plate"] for r in rows])


@bp.route("/heartbeat", methods=["POST"])
def heartbeat():
    data = request.get_json(silent=True) or {}
    db.set_setting("heartbeat_at", utils.now_str())
    db.set_setting("heartbeat_info", str(data.get("info", ""))[:200])
    return jsonify(ok=True)


@bp.route("/violations", methods=["POST"])
def create_violation():
    # Accepts multipart form data (with an image) or JSON (without one).
    data = request.form if request.form else (request.get_json(silent=True) or {})

    vtype = str(data.get("type", "")).strip().lower()
    if vtype not in ("speeding", "honking"):
        return _fail("type must be 'speeding' or 'honking'.", 400)

    try:
        speed = _number(data["speed_kmh"], "speed_kmh", 0, 250) if vtype == "speeding" else None
        db_level = _number(data["db_level"], "db_level", 0, 200) if data.get("db_level") not in (None, "") else None
        confidence = _number(data["plate_confidence"], "plate_confidence", 0, 1) \
            if data.get("plate_confidence") not in (None, "") else None
        detected_at = utils.parse_dt(data.get("detected_at"))
    except KeyError:
        return _fail("speed_kmh is required for speeding violations.", 400)
    except ValueError as exc:
        return _fail(str(exc), 400)

    plate = utils.normalize_plate(str(data.get("plate", "")))[:12] or None
    vclass = str(data.get("vehicle_class", "")).strip().lower()
    vclass = vclass if vclass in utils.VEHICLE_CLASSES else None
    sound_note = str(data.get("sound_note", "")).strip()[:200] or None   # what the sound model heard
    conn = db.get_db()

    # Guard against the detector reporting the same event repeatedly.
    if plate:
        dup = conn.execute(
            "SELECT id, record_id FROM violations WHERE type = ? AND plate = ? "
            "AND detected_at >= datetime(?, '-30 seconds') AND detected_at <= datetime(?, '+30 seconds')",
            (vtype, plate, detected_at, detected_at)).fetchone()
    else:
        dup = conn.execute(
            "SELECT id, record_id FROM violations WHERE type = ? AND plate IS NULL "
            "AND detected_at >= datetime(?, '-5 seconds') AND detected_at <= datetime(?, '+5 seconds')",
            (vtype, detected_at, detected_at)).fetchone()
    if dup:
        return jsonify(duplicate=True, id=dup["id"], record_id=dup["record_id"]), 200

    image_name = None
    upload = request.files.get("image")
    if upload:
        raw = upload.read()
        ext = utils.image_extension(raw)
        if not ext:
            return _fail("image must be a JPEG or PNG file.", 400)
        image_name = f"{uuid.uuid4().hex}.{ext}"
        with open(os.path.join(current_app.config["EVIDENCE_DIR"], image_name), "wb") as fh:
            fh.write(raw)

    video_name = None
    clip = request.files.get("video")
    if clip:
        raw = clip.read()
        if utils.is_mp4(raw):
            video_name = f"{uuid.uuid4().hex}.mp4"
            with open(os.path.join(current_app.config["EVIDENCE_DIR"], video_name), "wb") as fh:
                fh.write(raw)
        # a clip that is not a valid MP4 is dropped; the violation is still recorded with its photo

    audio_name = None
    sound = request.files.get("audio")
    if sound:
        raw = sound.read()
        if utils.is_wav(raw):
            audio_name = f"{uuid.uuid4().hex}.wav"
            with open(os.path.join(current_app.config["EVIDENCE_DIR"], audio_name), "wb") as fh:
                fh.write(raw)

    # Only verified vehicles are linked to a driver.
    vehicle = None
    if plate:
        vehicle = conn.execute(
            "SELECT id, user_id FROM vehicles WHERE plate = ? AND verified = 1", (plate,)
        ).fetchone()

    now = utils.now_str()
    cur = conn.execute(
        "INSERT INTO violations(type, plate, plate_confidence, vehicle_class, vehicle_id, user_id, speed_kmh, "
        "speed_limit, db_level, image_path, video_path, audio_path, detected_at, created_at, sound_note) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (vtype, plate, confidence, vclass, vehicle["id"] if vehicle else None,
         vehicle["user_id"] if vehicle else None, speed,
         float(db.get_setting("speed_limit", "10")) if vtype == "speeding" else None,
         db_level, image_name, video_name, audio_name, detected_at, now, sound_note))
    vid = cur.lastrowid
    record_id = f"RS-{detected_at[:10].replace('-', '')}-{vid:05d}"
    conn.execute("UPDATE violations SET record_id = ? WHERE id = ?", (record_id, vid))
    conn.commit()

    sms_status = "not_registered"
    offense_no = None
    held = vtype == "honking" and db.get_setting("horn_needs_review", "1") == "1"
    if not vehicle:
        db.recompute_plate_offenses(plate)     # unregistered: traced by plate, or counted alone if unread
        offense_no = conn.execute("SELECT offense_no FROM violations WHERE id = ?", (vid,)).fetchone()[0]
    if vehicle:
        db.recompute_offenses(vehicle["user_id"])
        offense_no = conn.execute("SELECT offense_no FROM violations WHERE id = ?", (vid,)).fetchone()[0]
        owner = conn.execute("SELECT phone FROM users WHERE id = ?", (vehicle["user_id"],)).fetchone()
        if held:
            # one microphone cannot tell which vehicle honked: an administrator checks the clip first
            sms_status = "awaiting_review"
        elif owner and owner["phone"]:
            message = sms.compose(vtype, plate, detected_at, offense_no)
            sms_status = sms.send(vid, owner["phone"], message)
        else:
            sms_status = "no_phone"
        conn.execute("UPDATE violations SET sms_status = ? WHERE id = ?", (sms_status, vid))
        conn.commit()

    if held and not vehicle:
        sms_status = "awaiting_review"
        conn.execute("UPDATE violations SET sms_status = ? WHERE id = ?", (sms_status, vid))
        conn.commit()

    return jsonify(id=vid, record_id=record_id, registered=bool(vehicle),
                   offense_no=offense_no, sms=sms_status), 201
