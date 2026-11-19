"""Shared helpers: time, validation, CSRF, access control, template filters."""
import functools
import re
import secrets
from datetime import datetime, timedelta, timezone

from flask import abort, g, redirect, request, session, url_for

from . import db

PH_TZ = timezone(timedelta(hours=8))  # Philippines has no daylight saving
FMT = "%Y-%m-%d %H:%M:%S"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def now_str():
    return datetime.now(PH_TZ).strftime(FMT)


def today_str():
    return now_str()[:10]


def last_days(n):
    today = datetime.now(PH_TZ).date()
    return [(today - timedelta(days=i)).isoformat() for i in range(n - 1, -1, -1)]


def parse_dt(value):
    """Accept an ISO 8601 string from the detector and return local time text."""
    if not value:
        return now_str()
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("detected_at must be an ISO 8601 date and time")
    if dt.tzinfo:
        dt = dt.astimezone(PH_TZ)
    return dt.strftime(FMT)


def fmt_dt(value):
    if not value:
        return ""
    try:
        dt = datetime.strptime(value, FMT)
    except ValueError:
        return value
    return f"{dt.day} {dt:%b %Y}, {dt.strftime('%I:%M %p').lstrip('0')}"


def normalize_plate(text):
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


VEHICLE_CLASSES = ("car", "van", "jeepney", "tricycle", "motorcycle", "bus", "truck", "multicab")

# Types a driver or the office can register. Matches what the detector reports, plus "other".
VEHICLE_TYPES = VEHICLE_CLASSES + ("other",)
VEHICLE_TYPE_CHOICES = [(t, t.capitalize()) for t in VEHICLE_TYPES]


def clean_vehicle_type(value):
    """Return a known vehicle type, or "car" when the form sent something else."""
    value = (value or "").strip().lower()
    return value if value in VEHICLE_TYPES else "car"


def format_plate(plate):
    """How a plate is shown. It is stored without spaces so it can be matched, but a car plate is
    printed with a space (ABC 1234) and a motorcycle plate without one (1234AB). A temporary plate of
    ten digits is shown as 1201-401686. Anything else is shown as it is."""
    plate = plate or ""
    if re.fullmatch(r"[A-Z]{3}\d{3,4}", plate):
        return f"{plate[:3]} {plate[3:]}"
    if re.fullmatch(r"\d{10}", plate):
        return f"{plate[:4]}-{plate[4:]}"
    return plate


def valid_plate(plate):
    return bool(5 <= len(plate) <= 8 and re.search(r"[A-Z]", plate) and re.search(r"\d", plate))


def normalize_phone(text):
    digits = re.sub(r"[\s\-()]", "", text or "")
    m = re.fullmatch(r"(?:\+?63|0)(9\d{9})", digits)
    return "0" + m.group(1) if m else None


def next_step(offense_no):
    """What the campus rules require after the nth offense."""
    if not offense_no:
        return ""
    if offense_no == 1:
        return "Submit an appeal or a letter of apology."
    if offense_no == 2:
        return "Report in person to the Security Services office."
    return "Your gate pass is suspended."


def image_extension(data):
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return None


def is_mp4(data):
    return len(data) > 12 and data[4:8] == b"ftyp"


def is_wav(data):
    return len(data) > 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_urlsafe(32)
    return session["_csrf"]


def csrf_protect():
    if request.method in ("POST", "PUT", "PATCH", "DELETE") and request.blueprint != "api":
        sent = request.form.get("csrf_token", "")
        expected = session.get("_csrf", "")
        if not sent or not expected or not secrets.compare_digest(sent, expected):
            abort(400, "Your session expired. Reload the page and try again.")


def load_user():
    g.user = None
    uid = session.get("uid")
    if uid:
        row = db.get_db().execute(
            "SELECT * FROM users WHERE id = ? AND active = 1", (uid,)
        ).fetchone()
        if row is None:
            session.clear()
        g.user = row


def home_for(user):
    return url_for("admin.dashboard" if user["role"] == "admin" else "user.dashboard")


def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            return redirect(url_for("auth.login"))
        return view(*args, **kwargs)

    return wrapped


def site_url():
    """Configured production base URL (ROADSENSE_SITE_URL) or the current request origin."""
    from flask import current_app, request
    configured = (current_app.config.get("SITE_URL") or "").rstrip("/")
    return configured or request.url_root.rstrip("/")


def inject_globals():
    from flask import request
    base = site_url()
    path = "/" if request.path == "/" else request.path.rstrip("/")
    return {
        "csrf_token": csrf_token,
        "site_url": base,
        "canonical_url": f"{base}{path}",
        "vehicle_type_choices": VEHICLE_TYPE_CHOICES,
    }


LABELS = {
    "flagged": "Flagged",
    "under_review": "Under review",
    "resolved": "Resolved",
    "dismissed": "Dismissed",
    "speeding": "Speeding",
    "honking": "Honking",
}


def label(value):
    return LABELS.get(value, value)
