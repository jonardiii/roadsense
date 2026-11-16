"""SMS notifications through the Semaphore gateway.

sms_mode setting:
  off  - send nothing
  log  - dry run: write the message to the SMS log only (default, safe for testing)
  live - send through Semaphore
"""
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from . import db, utils

SEMAPHORE_URL = "https://api.semaphore.co/api/v4/messages"


def compose(violation_type, plate, detected_at, offense_no):
    what = "Speeding" if violation_type == "speeding" else "Horn"
    step = utils.next_step(offense_no)
    return (
        f"RoadSense: {what} violation for {utils.format_plate(plate) if plate else 'your vehicle'} "
        f"on {utils.fmt_dt(detected_at)}. Offense #{offense_no}. {step} "
        "Log in to view the photo and appeal."
    )


MAX_ATTEMPTS = 30           # about half an hour of tries at one a minute, then the SMS is marked failed


def _deliver(phone, message):
    """One attempt through Semaphore. Returns (status, detail): sent, queued (worth trying again
    later: no internet, timeout, gateway down) or failed (will not work if retried: no key, rejected)."""
    key = os.environ.get("SEMAPHORE_API_KEY") or db.get_setting("semaphore_api_key")
    if not key:
        return "failed", "No Semaphore API key is set"
    payload = {"apikey": key, "number": phone, "message": message}
    sender = os.environ.get("SEMAPHORE_SENDER") or db.get_setting("semaphore_sender")
    if sender:
        payload["sendername"] = sender
    req = Request(SEMAPHORE_URL, data=urlencode(payload).encode(), method="POST")
    try:
        with urlopen(req, timeout=10) as resp:
            return "sent", resp.read().decode()[:300]
    except HTTPError as exc:
        # A busy or broken gateway may recover; a rejected request (wrong key, bad number) will not.
        return ("queued" if exc.code >= 500 or exc.code == 429 else "failed"), f"Gateway returned HTTP {exc.code}"
    except (URLError, TimeoutError, OSError) as exc:
        return "queued", f"Could not reach gateway: {exc}"


def send(violation_id, phone, message):
    """Send or log one message. Returns a short status string.

    In live mode a message that could not be sent because the internet or the gateway was
    unavailable is kept as 'queued' and tried again by retry_pending()."""
    mode = db.get_setting("sms_mode", "log")
    conn = db.get_db()
    status, detail = "skipped", "SMS is turned off"
    if mode == "log":
        status, detail = "logged", "Dry run: not sent"
    elif mode == "live":
        status, detail = _deliver(phone, message)
    conn.execute(
        "INSERT INTO sms_log(violation_id, phone, message, status, detail, attempts, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (violation_id, phone, message, status, detail, 1 if mode == "live" else 0, utils.now_str()),
    )
    conn.commit()
    return status


def retry_one(row):
    """Try a queued or failed message again now. Updates its log row and the violation. Returns the status."""
    conn = db.get_db()
    if db.get_setting("sms_mode", "log") != "live":
        return row["status"]
    status, detail = _deliver(row["phone"], row["message"])
    attempts = (row["attempts"] or 0) + 1
    if status == "queued" and attempts >= MAX_ATTEMPTS:
        status, detail = "failed", f"Gave up after {attempts} attempts. {detail}"
    conn.execute("UPDATE sms_log SET status = ?, detail = ?, attempts = ? WHERE id = ?",
                 (status, detail, attempts, row["id"]))
    if row["violation_id"]:
        conn.execute("UPDATE violations SET sms_status = ? WHERE id = ?", (status, row["violation_id"]))
    conn.commit()
    return status


def retry_pending():
    """Try every queued message once, oldest first. Stops at the first one that still cannot be
    sent because there is no internet, so a dead connection costs one timeout and not one per message."""
    conn = db.get_db()
    rows = conn.execute("SELECT * FROM sms_log WHERE status = 'queued' ORDER BY id").fetchall()
    done = 0
    for row in rows:
        status = retry_one(row)
        if status == "queued":
            break
        done += 1
    return done


def start_retry_thread(app, every=60):
    """Background loop that retries queued messages while the website is running."""
    import threading
    import time

    def loop():
        while True:
            time.sleep(every)
            try:
                with app.app_context():
                    retry_pending()
            except Exception:       # never let a bad row stop the loop
                app.logger.exception("SMS retry failed")

    threading.Thread(target=loop, name="sms-retry", daemon=True).start()
