"""Administrator pages: dashboard, violations, reports, drivers, vehicles, settings."""
import csv
import io
import os
import secrets
from datetime import datetime, timedelta

from flask import (Blueprint, Response, abort, current_app, flash, g, redirect,
                   render_template, request, send_from_directory, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from . import db, sms, utils

bp = Blueprint("admin", __name__, url_prefix="/admin")

PER_PAGE = 20


@bp.before_request
def _guard():
    if g.user is None:
        return redirect(url_for("auth.login"))
    if g.user["role"] != "admin":
        abort(403)


@bp.route("/")
def dashboard():
    conn = db.get_db()
    today = utils.today_str()
    valid = "status != 'dismissed'"
    stats = {
        "today": conn.execute(
            f"SELECT COUNT(*) FROM violations WHERE substr(detected_at,1,10) = ? AND {valid}",
            (today,)).fetchone()[0],
        "drivers": conn.execute("SELECT COUNT(*) FROM users WHERE role = 'user'").fetchone()[0],
        "total": conn.execute(f"SELECT COUNT(*) FROM violations WHERE {valid}").fetchone()[0],
        "appeals": conn.execute("SELECT COUNT(*) FROM appeals WHERE status = 'pending'").fetchone()[0],
        "unmatched": conn.execute(
            f"SELECT COUNT(*) FROM violations WHERE user_id IS NULL AND {valid}").fetchone()[0],
        "unverified": conn.execute("SELECT COUNT(*) FROM vehicles WHERE verified = 0").fetchone()[0],
    }
    days = utils.last_days(7)
    rows = conn.execute(
        f"SELECT substr(detected_at,1,10) AS day, type, COUNT(*) AS c FROM violations "
        f"WHERE substr(detected_at,1,10) >= ? AND {valid} GROUP BY day, type", (days[0],)).fetchall()
    series = {d: {"speeding": 0, "honking": 0} for d in days}
    for r in rows:
        if r["day"] in series:
            series[r["day"]][r["type"]] = r["c"]
    peak = max([sum(v.values()) for v in series.values()] + [1])
    chart = [{
        "day": datetime.strptime(d, "%Y-%m-%d").strftime("%a"),
        "date": d,
        "speeding": v["speeding"], "honking": v["honking"],
        "total": v["speeding"] + v["honking"],
        "height": round(100 * (v["speeding"] + v["honking"]) / peak),
        "speeding_h": round(100 * v["speeding"] / max(v["speeding"] + v["honking"], 1)),
    } for d, v in series.items()]
    recent = conn.execute(
        "SELECT v.*, u.full_name FROM violations v LEFT JOIN users u ON u.id = v.user_id "
        "ORDER BY v.detected_at DESC, v.id DESC LIMIT 8").fetchall()
    beat = db.get_setting("heartbeat_at")
    return render_template("admin/dashboard.html", stats=stats, chart=chart, recent=recent, beat=beat)


def _violation_filters(args):
    where, params = [], []
    plate = utils.normalize_plate(args.get("q", ""))
    if plate:
        where.append("v.plate LIKE ?")
        params.append(f"%{plate}%")
    if args.get("type") in ("speeding", "honking"):
        where.append("v.type = ?")
        params.append(args["type"])
    if args.get("status") in ("flagged", "under_review", "resolved", "dismissed"):
        where.append("v.status = ?")
        params.append(args["status"])
    if utils.DATE_RE.match(args.get("date_from", "")):
        where.append("substr(v.detected_at,1,10) >= ?")
        params.append(args["date_from"])
    if utils.DATE_RE.match(args.get("date_to", "")):
        where.append("substr(v.detected_at,1,10) <= ?")
        params.append(args["date_to"])
    return ("WHERE " + " AND ".join(where)) if where else "", params


@bp.route("/violations")
def violations():
    conn = db.get_db()
    clause, params = _violation_filters(request.args)
    total = conn.execute(f"SELECT COUNT(*) FROM violations v {clause}", params).fetchone()[0]
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    rows = conn.execute(
        f"SELECT v.*, u.full_name FROM violations v LEFT JOIN users u ON u.id = v.user_id {clause} "
        "ORDER BY v.detected_at DESC, v.id DESC LIMIT ? OFFSET ?",
        params + [PER_PAGE, (page - 1) * PER_PAGE]).fetchall()
    pages = max(1, -(-total // PER_PAGE))
    args = {k: v for k, v in request.args.items() if k != "page" and v}
    return render_template("admin/violations.html", rows=rows, total=total, page=page,
                           pages=pages, args=args)


@bp.route("/violations/<int:vid>")
def violation_detail(vid):
    conn = db.get_db()
    v = conn.execute(
        "SELECT v.*, u.full_name, u.email, u.phone FROM violations v "
        "LEFT JOIN users u ON u.id = v.user_id WHERE v.id = ?", (vid,)).fetchone()
    if v is None:
        abort(404)
    appeals = conn.execute("SELECT * FROM appeals WHERE violation_id = ? ORDER BY id", (vid,)).fetchall()
    sms_rows = conn.execute("SELECT * FROM sms_log WHERE violation_id = ? ORDER BY id", (vid,)).fetchall()
    pending = any(a["status"] == "pending" for a in appeals)
    awaiting = v["type"] == "honking" and v["sms_status"] == "awaiting_review" and v["status"] == "flagged"
    can_review = v["status"] in ("flagged", "under_review") and (
        pending or awaiting or db.get_setting("allow_direct_review") == "1")
    return render_template("admin/violation.html", v=v, appeals=appeals, sms_rows=sms_rows,
                           can_review=can_review, pending=pending, awaiting=awaiting)


def _remove_violations(ids):
    """Delete violations with their appeals, text-message log and photo files. Returns how many."""
    conn = db.get_db()
    ids = list(ids)
    users, plates, photos = set(), set(), []
    for i in range(0, len(ids), 500):
        part = ids[i:i + 500]
        marks = ",".join("?" * len(part))
        for row in conn.execute(f"SELECT user_id, plate, image_path, video_path, audio_path FROM violations WHERE id IN ({marks})", part):
            if row["user_id"] is not None:
                users.add(row["user_id"])
            else:
                plates.add(row["plate"])
            if row["image_path"]:
                photos.append(row["image_path"])
            if row["video_path"]:
                photos.append(row["video_path"])
            if row["audio_path"]:
                photos.append(row["audio_path"])
        conn.execute(f"DELETE FROM sms_log WHERE violation_id IN ({marks})", part)
        conn.execute(f"DELETE FROM appeals WHERE violation_id IN ({marks})", part)
        conn.execute(f"DELETE FROM violations WHERE id IN ({marks})", part)
    conn.commit()
    folder = current_app.config["EVIDENCE_DIR"]
    for name in photos:
        path = os.path.join(folder, os.path.basename(name))   # never leave the evidence folder
        try:
            os.remove(path)
        except OSError:
            pass
    for uid in users:
        db.recompute_offenses(uid)   # offense numbers close up after a delete
    for plate in plates:
        db.recompute_plate_offenses(plate)   # a None plate renumbers the unidentified ones
    return len(ids)


@bp.route("/violations/<int:vid>/delete", methods=["POST"])
def delete_violation(vid):
    row = db.get_db().execute("SELECT record_id FROM violations WHERE id = ?", (vid,)).fetchone()
    if row is None:
        abort(404)
    _remove_violations([vid])
    flash(f"Violation {row['record_id'] or vid} deleted.", "ok")
    return redirect(url_for("admin.violations"))


@bp.route("/violations/delete-matching", methods=["POST"])
def delete_matching_violations():
    """Delete every violation that matches the filters on the list page (all of them with no filter)."""
    if request.form.get("confirm", "").strip() != "DELETE":
        flash("Nothing was deleted. Type DELETE in capital letters to confirm.", "error")
        return redirect(url_for("admin.violations"))
    clause, params = _violation_filters(request.form)
    ids = [r["id"] for r in db.get_db().execute(f"SELECT v.id FROM violations v {clause}", params)]
    n = _remove_violations(ids)
    flash(f"{n} violation{'' if n == 1 else 's'} deleted.", "ok")
    return redirect(url_for("admin.violations"))


@bp.route("/violations/<int:vid>/resend-sms", methods=["POST"])
def resend_sms(vid):
    conn = db.get_db()
    row = conn.execute("SELECT * FROM sms_log WHERE violation_id = ? AND status IN ('queued', 'failed') "
                       "ORDER BY id DESC LIMIT 1", (vid,)).fetchone()
    if row is None:
        flash("There is no unsent text message for this violation.", "error")
    elif db.get_setting("sms_mode", "log") != "live":
        flash("SMS is not in live mode, so nothing can be sent. Change it on the Settings page.", "error")
    else:
        status = sms.retry_one({**dict(row), "attempts": 0})   # an admin's try starts the count again
        flash({"sent": "The text message was sent.",
               "queued": "Could not reach the SMS service. It will be tried again automatically.",
               }.get(status, "The text message could not be sent. See the SMS log for the reason."),
              "ok" if status == "sent" else "error")
    return redirect(url_for("admin.violation_detail", vid=vid))


@bp.route("/violations/<int:vid>/decision", methods=["POST"])
def decide(vid):
    conn = db.get_db()
    v = conn.execute("SELECT * FROM violations WHERE id = ?", (vid,)).fetchone()
    if v is None:
        abort(404)
    decision = request.form.get("decision")
    note = request.form.get("note", "").strip()[:500]
    if decision not in ("resolved", "dismissed"):
        abort(400)
    pending = conn.execute(
        "SELECT 1 FROM appeals WHERE violation_id = ? AND status = 'pending'", (vid,)).fetchone()
    awaiting = v["type"] == "honking" and v["sms_status"] == "awaiting_review"
    if v["status"] not in ("flagged", "under_review") or not (
            pending or awaiting or db.get_setting("allow_direct_review") == "1"):
        flash("This violation can only be reviewed after the driver submits an appeal.", "error")
        return redirect(url_for("admin.violation_detail", vid=vid))
    conn.execute("UPDATE violations SET status = ? WHERE id = ?", (decision, vid))
    conn.execute(
        "UPDATE appeals SET status = 'reviewed', admin_note = ?, reviewed_at = ? "
        "WHERE violation_id = ? AND status = 'pending'", (note or None, utils.now_str(), vid))
    conn.commit()
    db.recompute_offenses(v["user_id"])
    if v["user_id"] is None:
        db.recompute_plate_offenses(v["plate"])
    flash("Violation marked as " + utils.label(decision).lower() + ".", "ok")
    return redirect(url_for("admin.violation_detail", vid=vid))


@bp.route("/violations/<int:vid>/confirm-horn", methods=["POST"])
def confirm_horn(vid):
    """The administrator watched the clip and agrees the vehicle in it honked: now the driver is texted."""
    conn = db.get_db()
    v = conn.execute("SELECT * FROM violations WHERE id = ?", (vid,)).fetchone()
    if v is None:
        abort(404)
    if v["type"] != "honking" or v["sms_status"] != "awaiting_review":
        flash("This record is not waiting for a horn review.", "error")
        return redirect(url_for("admin.violation_detail", vid=vid))
    owner = conn.execute("SELECT phone FROM users WHERE id = ?", (v["user_id"],)).fetchone() if v["user_id"] else None
    if v["user_id"] is None:
        status = "not_registered"
    elif owner and owner["phone"]:
        status = sms.send(vid, owner["phone"], sms.compose("honking", v["plate"], v["detected_at"], v["offense_no"]))
    else:
        status = "no_phone"
    conn.execute("UPDATE violations SET sms_status = ? WHERE id = ?", (status, vid))
    conn.commit()
    flash("Horn confirmed." + (" The driver was texted." if v["user_id"] else ""), "ok")
    return redirect(url_for("admin.violation_detail", vid=vid))


@bp.route("/violations/<int:vid>/set-plate", methods=["POST"])
def set_plate(vid):
    """The plate was not read, but the photo or video shows it. Enter it by hand."""
    conn = db.get_db()
    v = conn.execute("SELECT * FROM violations WHERE id = ?", (vid,)).fetchone()
    if v is None:
        abort(404)
    plate = utils.normalize_plate(request.form.get("plate", ""))
    if v["plate"]:
        flash("This record already has a plate.", "error")
    elif not utils.valid_plate(plate):
        flash("Check the plate number. Use letters and digits, like ABC1234.", "error")
    else:
        vehicle = conn.execute("SELECT id, user_id FROM vehicles WHERE plate = ? AND verified = 1",
                               (plate,)).fetchone()
        conn.execute("UPDATE violations SET plate = ?, plate_confidence = NULL, vehicle_id = ?, user_id = ? "
                     "WHERE id = ?", (plate, vehicle["id"] if vehicle else None,
                                      vehicle["user_id"] if vehicle else None, vid))
        conn.commit()
        if vehicle:
            db.recompute_offenses(vehicle["user_id"])
        else:
            db.recompute_plate_offenses(plate)
        db.recompute_plate_offenses(None)        # the unidentified ones close up
        if vehicle and v["type"] == "honking":
            conn.execute("UPDATE violations SET sms_status = 'awaiting_review' WHERE id = ?", (vid,))
            conn.commit()
        elif vehicle:
            owner = conn.execute("SELECT phone FROM users WHERE id = ?", (vehicle["user_id"],)).fetchone()
            offense = conn.execute("SELECT offense_no FROM violations WHERE id = ?", (vid,)).fetchone()[0]
            if owner and owner["phone"]:
                status = sms.send(vid, owner["phone"], sms.compose(v["type"], plate, v["detected_at"], offense))
            else:
                status = "no_phone"
            conn.execute("UPDATE violations SET sms_status = ? WHERE id = ?", (status, vid))
            conn.commit()
        flash(f"Plate set to {plate}. The record now counts under that plate.", "ok")
    return redirect(url_for("admin.violation_detail", vid=vid))


@bp.route("/repeat")
def repeat():
    """Plates that are not registered to a driver and keep violating, for the guards at the exit."""
    conn = db.get_db()
    rows = conn.execute(
        "SELECT plate, COUNT(*) AS n, MIN(detected_at) AS first_at, MAX(detected_at) AS last_at, "
        "SUM(type = 'speeding') AS speeding, SUM(type = 'honking') AS honking, MAX(vehicle_class) AS kind "
        "FROM violations WHERE user_id IS NULL AND plate IS NOT NULL AND status != 'dismissed' "
        "GROUP BY plate HAVING n >= 2 ORDER BY n DESC, last_at DESC").fetchall()
    unknown = conn.execute(
        "SELECT id, record_id, type, detected_at, vehicle_class FROM violations "
        "WHERE plate IS NULL AND user_id IS NULL AND status != 'dismissed' "
        "ORDER BY detected_at DESC LIMIT 30").fetchall()
    return render_template("admin/repeat.html", rows=rows, unknown=unknown)


@bp.route("/evidence/<int:vid>")
def evidence(vid):
    row = db.get_db().execute("SELECT image_path FROM violations WHERE id = ?", (vid,)).fetchone()
    if row is None or not row["image_path"]:
        abort(404)
    resp = send_from_directory(current_app.config["EVIDENCE_DIR"], row["image_path"])
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp


@bp.route("/clip/<int:vid>")
def clip(vid):
    row = db.get_db().execute("SELECT video_path FROM violations WHERE id = ?", (vid,)).fetchone()
    if row is None or not row["video_path"]:
        abort(404)
    resp = send_from_directory(current_app.config["EVIDENCE_DIR"], row["video_path"], mimetype="video/mp4")
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp


@bp.route("/sound/<int:vid>")
def sound(vid):
    row = db.get_db().execute("SELECT audio_path FROM violations WHERE id = ?", (vid,)).fetchone()
    if row is None or not row["audio_path"]:
        abort(404)
    resp = send_from_directory(current_app.config["EVIDENCE_DIR"], row["audio_path"], mimetype="audio/wav")
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp


@bp.route("/live")
def live():
    beat = db.get_setting("heartbeat_at")
    online = False
    if beat:
        try:
            age = datetime.strptime(utils.now_str(), utils.FMT) - datetime.strptime(beat, utils.FMT)
            online = age < timedelta(seconds=90)
        except ValueError:
            pass
    return render_template("admin/live.html", stream_url=db.get_setting("stream_url"),
                           beat=beat, online=online, info=db.get_setting("heartbeat_info"))


def _report_data(date_from, date_to):
    conn = db.get_db()
    base = "substr(detected_at,1,10) BETWEEN ? AND ? AND status != 'dismissed'"
    p = (date_from, date_to)
    by_type = {r["type"]: r["c"] for r in conn.execute(
        f"SELECT type, COUNT(*) c FROM violations WHERE {base} GROUP BY type", p)}
    by_day = conn.execute(
        f"SELECT substr(detected_at,1,10) day, SUM(type='speeding') s, SUM(type='honking') h "
        f"FROM violations WHERE {base} GROUP BY day ORDER BY day", p).fetchall()
    top = conn.execute(
        "SELECT v.plate, u.full_name, COUNT(*) c FROM violations v "
        "LEFT JOIN users u ON u.id = v.user_id "
        "WHERE substr(v.detected_at,1,10) BETWEEN ? AND ? AND v.status != 'dismissed' "
        "AND v.plate IS NOT NULL GROUP BY v.plate ORDER BY c DESC LIMIT 10", p).fetchall()
    return by_type, by_day, top


def _range():
    today = utils.today_str()
    start = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=29)).strftime("%Y-%m-%d")
    f = request.args.get("date_from", "")
    t = request.args.get("date_to", "")
    return (f if utils.DATE_RE.match(f) else start), (t if utils.DATE_RE.match(t) else today)


@bp.route("/reports")
def reports():
    date_from, date_to = _range()
    by_type, by_day, top = _report_data(date_from, date_to)
    return render_template("admin/reports.html", date_from=date_from, date_to=date_to,
                           by_type=by_type, by_day=by_day, top=top,
                           total=sum(by_type.values()))


def _safe_cell(value):
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


@bp.route("/reports.csv")
def reports_csv():
    date_from, date_to = _range()
    rows = db.get_db().execute(
        "SELECT v.record_id, v.detected_at, v.type, v.plate, u.full_name, v.speed_kmh, v.speed_limit, "
        "v.db_level, v.offense_no, v.status FROM violations v LEFT JOIN users u ON u.id = v.user_id "
        "WHERE substr(v.detected_at,1,10) BETWEEN ? AND ? ORDER BY v.detected_at",
        (date_from, date_to)).fetchall()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Record ID", "Detected at", "Type", "Plate", "Driver", "Speed (km/h)",
                     "Speed limit", "Horn level (dB)", "Offense no.", "Status"])
    for r in rows:
        writer.writerow([_safe_cell(r[k]) for k in r.keys()])
    return Response(buf.getvalue(), mimetype="text/csv", headers={
        "Content-Disposition": f"attachment; filename=roadsense_{date_from}_to_{date_to}.csv"})


@bp.route("/users")
def users():
    rows = db.get_db().execute(
        "SELECT u.*, (SELECT COUNT(*) FROM vehicles v WHERE v.user_id = u.id) AS vehicles, "
        "(SELECT COUNT(*) FROM violations x WHERE x.user_id = u.id AND x.status != 'dismissed') AS violations "
        "FROM users u ORDER BY u.role, u.full_name").fetchall()
    return render_template("admin/users.html", rows=rows)


@bp.route("/users/<int:uid>/toggle", methods=["POST"])
def toggle_user(uid):
    if uid == g.user["id"]:
        flash("You can't deactivate your own account.", "error")
        return redirect(url_for("admin.users"))
    conn = db.get_db()
    row = conn.execute("SELECT active FROM users WHERE id = ?", (uid,)).fetchone()
    if row is None:
        abort(404)
    conn.execute("UPDATE users SET active = ? WHERE id = ?", (0 if row["active"] else 1, uid))
    conn.commit()
    flash("Account " + ("deactivated." if row["active"] else "reactivated."), "ok")
    return redirect(url_for("admin.users"))


def _link_past(vehicle):
    conn = db.get_db()
    conn.execute(
        "UPDATE violations SET vehicle_id = ?, user_id = ? WHERE plate = ? AND vehicle_id IS NULL",
        (vehicle["id"], vehicle["user_id"], vehicle["plate"]))
    conn.commit()
    db.recompute_offenses(vehicle["user_id"])


def _unlink(vehicle):
    conn = db.get_db()
    conn.execute("UPDATE violations SET vehicle_id = NULL, user_id = NULL, offense_no = NULL "
                 "WHERE vehicle_id = ?", (vehicle["id"],))
    conn.commit()
    db.recompute_offenses(vehicle["user_id"])
    db.recompute_plate_offenses(vehicle["plate"])


@bp.route("/vehicles")
def vehicles():
    conn = db.get_db()
    rows = conn.execute(
        "SELECT v.*, u.full_name, u.gate_pass_no FROM vehicles v JOIN users u ON u.id = v.user_id "
        "ORDER BY v.verified, v.plate").fetchall()
    drivers = conn.execute("SELECT id, full_name FROM users WHERE role = 'user' AND active = 1 "
                           "ORDER BY full_name").fetchall()
    return render_template("admin/vehicles.html", rows=rows, drivers=drivers)


@bp.route("/vehicles/add", methods=["POST"])
def add_vehicle():
    conn = db.get_db()
    plate = utils.normalize_plate(request.form.get("plate", ""))
    vtype = utils.clean_vehicle_type(request.form.get("vehicle_type"))
    owner = conn.execute("SELECT id FROM users WHERE id = ? AND role = 'user'",
                         (request.form.get("user_id", 0),)).fetchone()
    if not owner or not utils.valid_plate(plate):
        flash("Pick a driver and enter a valid plate, like ABC1234.", "error")
    elif conn.execute("SELECT 1 FROM vehicles WHERE plate = ?", (plate,)).fetchone():
        flash("That plate is already registered.", "error")
    else:
        cur = conn.execute(
            "INSERT INTO vehicles(user_id, plate, vehicle_type, verified, created_at) VALUES (?, ?, ?, 1, ?)",
            (owner["id"], plate, vtype, utils.now_str()))
        conn.commit()
        _link_past(conn.execute("SELECT * FROM vehicles WHERE id = ?", (cur.lastrowid,)).fetchone())
        flash(f"{plate} added and verified.", "ok")
    return redirect(url_for("admin.vehicles"))


@bp.route("/vehicles/<int:vid>/verify", methods=["POST"])
def verify_vehicle(vid):
    conn = db.get_db()
    vehicle = conn.execute("SELECT * FROM vehicles WHERE id = ?", (vid,)).fetchone()
    if vehicle is None:
        abort(404)
    if vehicle["verified"]:
        conn.execute("UPDATE vehicles SET verified = 0 WHERE id = ?", (vid,))
        conn.commit()
        _unlink(vehicle)
        flash(f"{vehicle['plate']} is no longer verified.", "ok")
    else:
        conn.execute("UPDATE vehicles SET verified = 1 WHERE id = ?", (vid,))
        conn.commit()
        _link_past(vehicle)
        flash(f"{vehicle['plate']} verified. Past violations for this plate are now linked to the driver.", "ok")
    return redirect(url_for("admin.vehicles"))


@bp.route("/vehicles/<int:vid>/delete", methods=["POST"])
def delete_vehicle(vid):
    conn = db.get_db()
    vehicle = conn.execute("SELECT * FROM vehicles WHERE id = ?", (vid,)).fetchone()
    if vehicle is None:
        abort(404)
    _unlink(vehicle)
    conn.execute("DELETE FROM vehicles WHERE id = ?", (vid,))
    conn.commit()
    flash(f"{vehicle['plate']} removed.", "ok")
    return redirect(url_for("admin.vehicles"))


@bp.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        errors = []
        f = request.form

        def num(name, low, high):
            try:
                value = float(f.get(name, ""))
            except ValueError:
                errors.append(f"{name.replace('_', ' ').capitalize()} must be a number.")
                return None
            if not low <= value <= high:
                errors.append(f"{name.replace('_', ' ').capitalize()} must be between {low} and {high}.")
                return None
            return value

        speed = num("speed_limit", 1, 60)
        honk_db = num("honk_db", 40, 140)
        honk_s = num("honk_min_seconds", 0.1, 10)
        mode = f.get("sms_mode")
        if mode not in ("off", "log", "live"):
            errors.append("Choose an SMS mode.")
        sender = f.get("semaphore_sender", "").strip()
        if sender and not (sender.isalnum() and len(sender) <= 11):
            errors.append("The sender name can use up to 11 letters and digits.")
        stream = f.get("stream_url", "").strip()
        if stream and not stream.startswith(("http://", "https://")):
            errors.append("The live feed address must start with http:// or https://.")
        if errors:
            for e in errors:
                flash(e, "error")
        else:
            db.set_setting("speed_limit", f"{speed:g}")
            db.set_setting("honk_db", f"{honk_db:g}")
            db.set_setting("honk_min_seconds", f"{honk_s:g}")
            db.set_setting("sms_mode", mode)
            db.set_setting("semaphore_sender", sender)
            db.set_setting("stream_url", stream)
            db.set_setting("allow_direct_review", "1" if f.get("allow_direct_review") else "0")
            db.set_setting("horn_needs_review", "1" if f.get("horn_needs_review") else "0")
            if f.get("semaphore_api_key", "").strip():
                db.set_setting("semaphore_api_key", f["semaphore_api_key"].strip())
            flash("Settings saved.", "ok")
        return redirect(url_for("admin.settings"))
    keys = ("speed_limit", "honk_db", "honk_min_seconds", "sms_mode", "semaphore_sender",
            "stream_url", "allow_direct_review", "horn_needs_review", "api_key")
    s = {k: db.get_setting(k) for k in keys}
    s["has_semaphore_key"] = bool(db.get_setting("semaphore_api_key"))
    return render_template("admin/settings.html", s=s)


@bp.route("/settings/api-key", methods=["POST"])
def regenerate_key():
    db.set_setting("api_key", secrets.token_urlsafe(32))
    flash("New detector API key created. Update it in the detector program.", "ok")
    return redirect(url_for("admin.settings"))


@bp.route("/profile", methods=["GET", "POST"])
def profile():
    conn = db.get_db()
    if request.method == "POST":
        if request.form.get("action") == "password":
            current = request.form.get("current", "")
            new = request.form.get("new", "")
            if not check_password_hash(g.user["password_hash"], current):
                flash("Your current password is incorrect.", "error")
            elif len(new) < 8 or new != request.form.get("confirm", ""):
                flash("Use a new password of at least 8 characters, entered twice the same.", "error")
            else:
                conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                             (generate_password_hash(new), g.user["id"]))
                conn.commit()
                flash("Password changed.", "ok")
        else:
            name = request.form.get("full_name", "").strip()
            email = request.form.get("email", "").strip().lower()
            raw_phone = request.form.get("phone", "").strip()
            phone = utils.normalize_phone(raw_phone) if raw_phone else None
            if not 2 <= len(name) <= 80:
                flash("Enter your full name.", "error")
            elif not utils.EMAIL_RE.match(email) or len(email) > 120:
                flash("Enter a valid email address.", "error")
            elif raw_phone and not phone:
                flash("Enter a Philippine mobile number, like 09171234567.", "error")
            elif conn.execute("SELECT 1 FROM users WHERE email = ? AND id != ?", (email, g.user["id"])).fetchone():
                flash("Another account already uses this email.", "error")
            else:
                conn.execute("UPDATE users SET full_name = ?, email = ?, phone = ? WHERE id = ?",
                             (name, email, phone, g.user["id"]))
                conn.commit()
                flash("Profile saved.", "ok")
        return redirect(url_for("admin.profile"))
    return render_template("admin/profile.html")
