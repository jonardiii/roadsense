"""Landing page, sign up, log in, password reset and log out."""
import hashlib
import hmac
import secrets
import time

from flask import (Blueprint, Response, current_app, flash, g, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from . import db, mailer, utils

bp = Blueprint("auth", __name__)

MAX_FAILS = 5
WINDOW_SECONDS = 300
RESET_TTL_SECONDS = 600
RESET_MAX_ATTEMPTS = 5


def _resets():
    """Active password-reset tokens keyed by lowercase email."""
    return current_app.extensions.setdefault("password_resets", {})


def _code_digest(code):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _failures():
    """Failed log-in attempts per (ip, email): [count, first_attempt_time]. Kept per app."""
    return current_app.extensions.setdefault("login_failures", {})


def _locked(key):
    entry = _failures().get(key)
    if not entry:
        return False
    if time.time() - entry[1] > WINDOW_SECONDS:
        _failures().pop(key, None)
        return False
    return entry[0] >= MAX_FAILS


@bp.route("/")
def landing():
    if g.user:
        return redirect(utils.home_for(g.user))
    return render_template("auth/landing.html")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(utils.home_for(g.user))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        key = (request.remote_addr, email)
        if _locked(key):
            flash("Too many failed attempts. Wait a few minutes and try again.", "error")
            return render_template("auth/login.html", email=email), 429
        row = db.get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row and row["active"] and check_password_hash(row["password_hash"], password):
            _failures().pop(key, None)
            session.clear()
            session["uid"] = row["id"]
            return redirect(utils.home_for(row))
        entry = _failures().setdefault(key, [0, time.time()])
        entry[0] += 1
        flash("The email or password is incorrect.", "error")
        return render_template("auth/login.html", email=email), 401
    return render_template("auth/login.html", email="")


@bp.route("/signup", methods=["GET", "POST"])
def signup():
    if g.user:
        return redirect(utils.home_for(g.user))
    form = {}
    if request.method == "POST":
        form = {k: request.form.get(k, "").strip() for k in
                ("full_name", "email", "phone", "gate_pass_no", "plate", "vehicle_type")}
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        errors = []

        if not 2 <= len(form["full_name"]) <= 80:
            errors.append("Enter your full name.")
        if not utils.EMAIL_RE.match(form["email"]) or len(form["email"]) > 120:
            errors.append("Enter a valid email address.")
        phone = utils.normalize_phone(form["phone"])
        if not phone:
            errors.append("Enter a Philippine mobile number, like 09171234567. SMS alerts go here.")
        if len(password) < 8:
            errors.append("Use a password with at least 8 characters.")
        if password != confirm:
            errors.append("The two passwords don't match.")
        if len(form["gate_pass_no"]) > 30:
            errors.append("The gate pass number is too long.")

        plate = utils.normalize_plate(form["plate"])
        if plate and not utils.valid_plate(plate):
            errors.append("Check the plate number. Use letters and digits, like ABC1234.")
        vtype = utils.clean_vehicle_type(form["vehicle_type"])

        conn = db.get_db()
        if not errors:
            if conn.execute("SELECT 1 FROM users WHERE email = ?", (form["email"],)).fetchone():
                errors.append("An account with this email already exists. Try logging in.")
            if plate and conn.execute("SELECT 1 FROM vehicles WHERE plate = ?", (plate,)).fetchone():
                errors.append("This plate is already registered. Ask the Security Services office if it's yours.")

        if errors:
            for msg in errors:
                flash(msg, "error")
            return render_template("auth/signup.html", form=form), 400

        now = utils.now_str()
        cur = conn.execute(
            "INSERT INTO users(full_name, email, phone, gate_pass_no, password_hash, role, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'user', ?)",
            (form["full_name"], form["email"].lower(), phone, form["gate_pass_no"] or None,
             generate_password_hash(password), now),
        )
        if plate:
            conn.execute(
                "INSERT INTO vehicles(user_id, plate, vehicle_type, verified, created_at) "
                "VALUES (?, ?, ?, 0, ?)", (cur.lastrowid, plate, vtype, now))
        conn.commit()
        session.clear()
        session["uid"] = cur.lastrowid
        flash("Account created. Violations are linked to you once the Security Services "
              "office verifies your plate.", "ok")
        return redirect(url_for("user.dashboard"))
    return render_template("auth/signup.html", form=form)


@bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if g.user:
        return redirect(utils.home_for(g.user))

    if request.method == "POST":
        action = request.form.get("action", "send_code")
        email = request.form.get("email", "").strip().lower()

        if action == "send_code":
            if not utils.EMAIL_RE.match(email) or len(email) > 120:
                flash("Enter a valid email address.", "error")
                return render_template("auth/forgot_password.html", step="email", email=email), 400

            row = db.get_db().execute(
                "SELECT id, active FROM users WHERE email = ?", (email,)
            ).fetchone()
            if row and row["active"]:
                code = f"{secrets.randbelow(1_000_000):06d}"
                _resets()[email] = {
                    "hash": _code_digest(code),
                    "expires_at": time.time() + RESET_TTL_SECONDS,
                    "attempts": 0,
                    "verified": False,
                }
                if current_app.config.get("TESTING"):
                    current_app.extensions["last_reset_code"] = (email, code)
                try:
                    mailer.send_reset_code(email, code)
                except Exception:
                    current_app.logger.exception("Failed to send password reset email to %s", email)
                    flash("Could not send verification email right now. Try again later.", "error")
                    return render_template("auth/forgot_password.html", step="email", email=email), 500
                flash("If an account exists for that email, a 6-digit verification code has been sent.", "ok")
            else:
                flash("If an account exists for that email, a 6-digit verification code has been sent.", "ok")

            return render_template("auth/forgot_password.html", step="verify", email=email)

        if action == "verify_code":
            code = request.form.get("code", "").strip()
            if not utils.EMAIL_RE.match(email):
                flash("Enter a valid email address.", "error")
                return render_template("auth/forgot_password.html", step="email", email=email), 400
            if not code.isdigit() or len(code) != 6:
                flash("Enter the 6-digit verification code sent to your email.", "error")
                return render_template("auth/forgot_password.html", step="verify", email=email), 400

            entry = _resets().get(email)
            if not entry or time.time() > entry["expires_at"]:
                _resets().pop(email, None)
                flash("The verification code has expired or is invalid. Request a new code.", "error")
                return render_template("auth/forgot_password.html", step="verify", email=email), 400
            if entry["attempts"] >= RESET_MAX_ATTEMPTS:
                _resets().pop(email, None)
                flash("Too many incorrect code attempts. Request a new code.", "error")
                return render_template("auth/forgot_password.html", step="verify", email=email), 400
            if not hmac.compare_digest(entry["hash"], _code_digest(code)):
                entry["attempts"] += 1
                flash("The verification code is incorrect.", "error")
                return render_template("auth/forgot_password.html", step="verify", email=email), 400

            entry["verified"] = True
            flash("Code verified. Enter your new password.", "ok")
            return render_template("auth/forgot_password.html", step="new_password", email=email)

        if action == "reset_password":
            password = request.form.get("password", "")
            confirm = request.form.get("confirm", "")
            entry = _resets().get(email)
            if not entry or not entry.get("verified") or time.time() > entry["expires_at"]:
                _resets().pop(email, None)
                flash("Your verification session has expired. Please request a new code.", "error")
                return render_template("auth/forgot_password.html", step="email", email=email), 400

            errors = []
            if len(password) < 8:
                errors.append("Use a password with at least 8 characters.")
            if password != confirm:
                errors.append("The two passwords don't match.")
            if errors:
                for msg in errors:
                    flash(msg, "error")
                return render_template("auth/forgot_password.html", step="new_password", email=email), 400

            conn = db.get_db()
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE email = ?",
                (generate_password_hash(password), email),
            )
            conn.commit()
            _resets().pop(email, None)
            for k in [k for k in _failures() if k[1] == email]:
                _failures().pop(k, None)
            flash("Your password has been updated. You can now sign in.", "ok")
            return redirect(url_for("auth.login"))

    return render_template("auth/forgot_password.html", step="email", email="")


@bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("auth.landing"))


@bp.route("/robots.txt")
def robots_txt():
    lines = [
        "User-agent: *",
        "Allow: /$",
        "Allow: /static/",
        "Disallow: /admin/",
        "Disallow: /me/",
        "Disallow: /api/",
        "Disallow: /login",
        "Disallow: /signup",
        "Disallow: /forgot-password",
        "Disallow: /logout",
        f"Sitemap: {utils.site_url()}/sitemap.xml",
    ]
    return Response("\n".join(lines) + "\n", mimetype="text/plain")


@bp.route("/sitemap.xml")
def sitemap_xml():
    base = utils.site_url()
    body = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
        f"  <url><loc>{base}/</loc><changefreq>weekly</changefreq><priority>1.0</priority></url>",
        "</urlset>",
    ]
    return Response("\n".join(body) + "\n", mimetype="application/xml")
