"""RoadSense web system: admin and driver portal plus the detector API."""
import os
import secrets

from flask import Flask, render_template

from . import db, utils


def _load_dotenv(path):
    """Load key=value pairs from a .env file into os.environ without overriding existing vars."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val


def create_app(test_config=None):
    if not test_config:
        _load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    app = Flask(__name__, instance_relative_config=True)
    os.makedirs(app.instance_path, exist_ok=True)

    secret = os.environ.get("ROADSENSE_SECRET")
    if not secret:
        key_file = os.path.join(app.instance_path, "secret_key")
        if not os.path.exists(key_file):
            with open(key_file, "w") as fh:
                fh.write(secrets.token_hex(32))
        with open(key_file) as fh:
            secret = fh.read().strip()

    app.config.update(
        SECRET_KEY=secret,
        DATABASE=os.path.join(app.instance_path, "roadsense.db"),
        EVIDENCE_DIR=os.path.join(app.instance_path, "evidence"),
        MAX_CONTENT_LENGTH=32 * 1024 * 1024,   # a photo plus a short video clip
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        TEMPLATES_AUTO_RELOAD=True,
        SITE_URL=os.environ.get("ROADSENSE_SITE_URL", "").rstrip("/"),
    )
    app.jinja_env.auto_reload = True
    if test_config:
        app.config.update(test_config)
    os.makedirs(app.config["EVIDENCE_DIR"], exist_ok=True)

    db.init_app(app)

    from . import admin, api, auth, cli, user

    app.register_blueprint(auth.bp)
    app.register_blueprint(admin.bp)
    app.register_blueprint(user.bp)
    app.register_blueprint(api.bp)
    cli.register(app)
    if app.config.get("SMS_RETRY_SECONDS", 60) and not app.config.get("TESTING"):
        from . import sms
        sms.start_retry_thread(app, app.config.get("SMS_RETRY_SECONDS", 60))

    app.before_request(utils.load_user)
    app.before_request(utils.csrf_protect)
    app.context_processor(utils.inject_globals)
    app.add_template_filter(utils.fmt_dt, "dt")
    app.add_template_filter(utils.label, "label")
    app.add_template_filter(utils.format_plate, "plate")
    app.add_template_filter(utils.next_step, "next_step")

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(413)
    def _error(err):
        titles = {
            400: "That request didn't work",
            403: "You don't have access to this page",
            404: "Page not found",
            413: "That file is too large",
        }
        return (
            render_template(
                "error.html", code=err.code, title=titles.get(err.code, "Error"),
                detail=getattr(err, "description", ""),
            ),
            err.code,
        )

    @app.after_request
    def _headers(resp):
        from flask import request
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        if request.path.startswith(("/admin", "/me", "/api", "/login", "/signup", "/forgot-password", "/logout")):
            resp.headers.setdefault("X-Robots-Tag", "noindex, nofollow")
        return resp

    return app
