"""SQLite access, schema, settings and offense numbering."""
import secrets
import sqlite3

from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  full_name TEXT NOT NULL,
  email TEXT NOT NULL UNIQUE COLLATE NOCASE,
  phone TEXT,
  gate_pass_no TEXT,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin','user')),
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vehicles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  plate TEXT NOT NULL UNIQUE,
  vehicle_type TEXT NOT NULL DEFAULT 'car',
  verified INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS violations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  record_id TEXT UNIQUE,
  type TEXT NOT NULL CHECK (type IN ('speeding','honking')),
  plate TEXT,
  plate_confidence REAL,
  vehicle_class TEXT,
  vehicle_id INTEGER REFERENCES vehicles(id) ON DELETE SET NULL,
  user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  speed_kmh REAL,
  speed_limit REAL,
  db_level REAL,
  image_path TEXT,
  video_path TEXT,
  audio_path TEXT,
  sound_note TEXT,
  detected_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'flagged'
    CHECK (status IN ('flagged','under_review','resolved','dismissed')),
  offense_no INTEGER,
  sms_status TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_viol_detected ON violations(detected_at);
CREATE INDEX IF NOT EXISTS idx_viol_user ON violations(user_id);
CREATE INDEX IF NOT EXISTS idx_viol_plate ON violations(plate);

CREATE TABLE IF NOT EXISTS appeals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  violation_id INTEGER NOT NULL REFERENCES violations(id) ON DELETE CASCADE,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK (kind IN ('apology','dispute')),
  message TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','reviewed')),
  admin_note TEXT,
  created_at TEXT NOT NULL,
  reviewed_at TEXT
);

CREATE TABLE IF NOT EXISTS sms_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  violation_id INTEGER REFERENCES violations(id) ON DELETE SET NULL,
  phone TEXT NOT NULL,
  message TEXT NOT NULL,
  status TEXT NOT NULL,
  detail TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""

DEFAULT_SETTINGS = {
    "speed_limit": "10",
    "honk_db": "85",
    "honk_min_seconds": "1",
    "sms_mode": "log",          # off | log (dry run) | live
    "semaphore_api_key": "",
    "semaphore_sender": "",
    "stream_url": "",
    "horn_needs_review": "1",    # 1 = no text message for a horn until an administrator confirms it
    "allow_direct_review": "0",  # 1 = admins may close a violation without an appeal
    "heartbeat_at": "",
    "heartbeat_info": "",
}


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(_err=None):
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def init_db():
    conn = get_db()
    conn.executescript(SCHEMA)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(violations)")}
    if "attempts" not in {r["name"] for r in conn.execute("PRAGMA table_info(sms_log)")}:
        conn.execute("ALTER TABLE sms_log ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")
    if "vehicle_class" not in cols:      # databases made before vehicle kinds were recorded
        conn.execute("ALTER TABLE violations ADD COLUMN vehicle_class TEXT")
    if "sound_note" not in cols:         # what the horn sound model heard
        conn.execute("ALTER TABLE violations ADD COLUMN sound_note TEXT")
    if "video_path" not in cols:         # databases made before video clips
        conn.execute("ALTER TABLE violations ADD COLUMN video_path TEXT")
    if "audio_path" not in cols:         # databases made before the separate sound clips
        conn.execute("ALTER TABLE violations ADD COLUMN audio_path TEXT")
    for key, value in DEFAULT_SETTINGS.items():
        conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)", (key, value))
    conn.execute(
        "INSERT OR IGNORE INTO settings(key, value) VALUES ('api_key', ?)",
        (secrets.token_urlsafe(32),),
    )
    conn.commit()


def init_app(app):
    app.teardown_appcontext(close_db)
    with app.app_context():
        init_db()


def get_setting(key, default=""):
    row = get_db().execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key, value):
    conn = get_db()
    conn.execute(
        "INSERT INTO settings(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )
    conn.commit()


def recompute_plate_offenses(plate):
    """Offense numbers for a plate that is not registered to any driver.

    The vehicle is still traced by its plate: its violations are numbered 1, 2, 3... in time order,
    so a repeat offender is known as one even before the owner registers. When the plate is later
    registered and verified, these violations move to the driver and are numbered there instead.
    """
    conn = get_db()
    if not plate:
        # Plate not read: the vehicle is not identified yet, but the violation is real and has its
        # photo and video, so it counts as an offense of its own until an administrator identifies it.
        conn.execute("UPDATE violations SET offense_no = CASE WHEN status = 'dismissed' THEN NULL ELSE 1 END "
                     "WHERE plate IS NULL AND user_id IS NULL")
        conn.commit()
        return
    rows = conn.execute(
        "SELECT id, status FROM violations WHERE plate = ? AND user_id IS NULL ORDER BY detected_at, id",
        (plate,),
    ).fetchall()
    n = 0
    for row in rows:
        if row["status"] == "dismissed":
            conn.execute("UPDATE violations SET offense_no = NULL WHERE id = ?", (row["id"],))
        else:
            n += 1
            conn.execute("UPDATE violations SET offense_no = ? WHERE id = ?", (n, row["id"]))
    conn.commit()


def recompute_offenses(user_id):
    """Number a driver's valid violations 1, 2, 3... in time order.

    Dismissed violations do not count toward the offense total.
    """
    if user_id is None:
        return
    conn = get_db()
    rows = conn.execute(
        "SELECT id, status FROM violations WHERE user_id = ? ORDER BY detected_at, id",
        (user_id,),
    ).fetchall()
    n = 0
    for row in rows:
        if row["status"] == "dismissed":
            conn.execute("UPDATE violations SET offense_no = NULL WHERE id = ?", (row["id"],))
        else:
            n += 1
            conn.execute("UPDATE violations SET offense_no = ? WHERE id = ?", (n, row["id"]))
    conn.commit()
