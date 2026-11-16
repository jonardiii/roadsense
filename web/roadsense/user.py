"""Driver pages: dashboard, own violations and appeals, vehicles, profile."""
from flask import (Blueprint, abort, current_app, flash, g, redirect, render_template,
                   request, send_from_directory, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from . import db, utils

bp = Blueprint("user", __name__, url_prefix="/me")


@bp.before_request
def _guard():
    if g.user is None:
        return redirect(url_for("auth.login"))
    if g.user["role"] == "admin":
        return redirect(url_for("admin.dashboard"))


@bp.route("/")
def dashboard():
    conn = db.get_db()
    uid = g.user["id"]
    total = conn.execute(
        "SELECT COUNT(*) FROM violations WHERE user_id = ? AND status != 'dismissed'", (uid,)).fetchone()[0]
    open_count = conn.execute(
        "SELECT COUNT(*) FROM violations WHERE user_id = ? AND status = 'flagged'", (uid,)).fetchone()[0]
    recent = conn.execute(
        "SELECT * FROM violations WHERE user_id = ? ORDER BY detected_at DESC, id DESC LIMIT 5",
        (uid,)).fetchall()
    unverified = conn.execute(
        "SELECT plate FROM vehicles WHERE user_id = ? AND verified = 0", (uid,)).fetchall()
    has_vehicle = conn.execute("SELECT 1 FROM vehicles WHERE user_id = ?", (uid,)).fetchone()
    overall = conn.execute(
        "SELECT COUNT(*) FROM violations WHERE status != 'dismissed'").fetchone()[0]
    return render_template("user/dashboard.html", total=total, open_count=open_count,
                           recent=recent, unverified=unverified, has_vehicle=bool(has_vehicle),
                           overall=overall)


@bp.route("/violations")
def violations():
    status = request.args.get("status", "")
    sql = "SELECT * FROM violations WHERE user_id = ?"
    params = [g.user["id"]]
    if status in ("flagged", "under_review", "resolved", "dismissed"):
        sql += " AND status = ?"
        params.append(status)
    rows = db.get_db().execute(sql + " ORDER BY detected_at DESC, id DESC", params).fetchall()
    return render_template("user/violations.html", rows=rows, status=status)


def _own_violation(vid):
    v = db.get_db().execute(
        "SELECT * FROM violations WHERE id = ? AND user_id = ?", (vid, g.user["id"])).fetchone()
    if v is None:
        abort(404)
    return v


@bp.route("/violations/<int:vid>")
def violation_detail(vid):
    v = _own_violation(vid)
    appeals = db.get_db().execute(
        "SELECT * FROM appeals WHERE violation_id = ? ORDER BY id", (vid,)).fetchall()
    return render_template("user/violation.html", v=v, appeals=appeals,
                           can_appeal=(v["status"] == "flagged" and not appeals))


@bp.route("/violations/<int:vid>/appeal", methods=["POST"])
def appeal(vid):
    v = _own_violation(vid)
    conn = db.get_db()
    kind = request.form.get("kind")
    message = request.form.get("message", "").strip()
    if v["status"] != "flagged" or conn.execute(
            "SELECT 1 FROM appeals WHERE violation_id = ?", (vid,)).fetchone():
        flash("This violation already has an appeal or is closed.", "error")
    elif kind not in ("apology", "dispute") or not 10 <= len(message) <= 1000:
        flash("Pick a type and write between 10 and 1000 characters.", "error")
    else:
        conn.execute(
            "INSERT INTO appeals(violation_id, user_id, kind, message, created_at) VALUES (?, ?, ?, ?, ?)",
            (vid, g.user["id"], kind, message, utils.now_str()))
        conn.execute("UPDATE violations SET status = 'under_review' WHERE id = ?", (vid,))
        conn.commit()
        flash("Your appeal was sent to the Security Services office.", "ok")
    return redirect(url_for("user.violation_detail", vid=vid))


@bp.route("/evidence/<int:vid>")
def evidence(vid):
    v = _own_violation(vid)
    if not v["image_path"]:
        abort(404)
    resp = send_from_directory(current_app.config["EVIDENCE_DIR"], v["image_path"])
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp


@bp.route("/clip/<int:vid>")
def clip(vid):
    v = _own_violation(vid)
    if not v["video_path"]:
        abort(404)
    resp = send_from_directory(current_app.config["EVIDENCE_DIR"], v["video_path"], mimetype="video/mp4")
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp


@bp.route("/sound/<int:vid>")
def sound(vid):
    v = _own_violation(vid)
    if not v["audio_path"]:
        abort(404)
    resp = send_from_directory(current_app.config["EVIDENCE_DIR"], v["audio_path"], mimetype="audio/wav")
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp


@bp.route("/vehicles", methods=["GET", "POST"])
def vehicles():
    conn = db.get_db()
    if request.method == "POST":
        plate = utils.normalize_plate(request.form.get("plate", ""))
        vtype = request.form.get("vehicle_type")
        vtype = vtype if vtype in ("car", "motorcycle", "other") else "car"
        if not utils.valid_plate(plate):
            flash("Check the plate number. Use letters and digits, like ABC1234.", "error")
        elif conn.execute("SELECT 1 FROM vehicles WHERE plate = ?", (plate,)).fetchone():
            flash("This plate is already registered. Ask the Security Services office if it's yours.", "error")
        else:
            conn.execute(
                "INSERT INTO vehicles(user_id, plate, vehicle_type, verified, created_at) VALUES (?, ?, ?, 0, ?)",
                (g.user["id"], plate, vtype, utils.now_str()))
            conn.commit()
            flash(f"{plate} added. It will show violations after the office verifies it.", "ok")
        return redirect(url_for("user.vehicles"))
    rows = conn.execute("SELECT * FROM vehicles WHERE user_id = ? ORDER BY plate", (g.user["id"],)).fetchall()
    return render_template("user/vehicles.html", rows=rows)


@bp.route("/vehicles/<int:vid>/remove", methods=["POST"])
def remove_vehicle(vid):
    conn = db.get_db()
    row = conn.execute("SELECT * FROM vehicles WHERE id = ? AND user_id = ?", (vid, g.user["id"])).fetchone()
    if row is None:
        abort(404)
    if row["verified"]:
        flash("Verified plates can only be removed by the Security Services office.", "error")
    else:
        conn.execute("DELETE FROM vehicles WHERE id = ?", (vid,))
        conn.commit()
        flash(f"{row['plate']} removed.", "ok")
    return redirect(url_for("user.vehicles"))


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
            phone = utils.normalize_phone(request.form.get("phone", ""))
            if not 2 <= len(name) <= 80 or not phone:
                flash("Enter your name and a Philippine mobile number, like 09171234567.", "error")
            else:
                conn.execute("UPDATE users SET full_name = ?, phone = ? WHERE id = ?",
                             (name, phone, g.user["id"]))
                conn.commit()
                flash("Profile saved.", "ok")
        return redirect(url_for("user.profile"))
    return render_template("user/profile.html")
