"""Command-line helpers:

  flask --app roadsense create-admin     create an administrator account
  flask --app roadsense seed-demo        add sample drivers and violations for screenshots
"""
import getpass
import random
from datetime import datetime, timedelta

import click
from werkzeug.security import generate_password_hash

from . import db, utils


def register(app):
    @app.cli.command("create-admin")
    @click.option("--email", prompt=True)
    @click.option("--name", prompt="Full name")
    def create_admin(email, name):
        password = getpass.getpass("Password (at least 8 characters): ")
        if len(password) < 8:
            raise click.ClickException("The password needs at least 8 characters.")
        conn = db.get_db()
        if conn.execute("SELECT 1 FROM users WHERE email = ?", (email.strip().lower(),)).fetchone():
            raise click.ClickException("An account with this email already exists.")
        conn.execute(
            "INSERT INTO users(full_name, email, password_hash, role, created_at) VALUES (?, ?, ?, 'admin', ?)",
            (name.strip(), email.strip().lower(), generate_password_hash(password), utils.now_str()))
        conn.commit()
        click.echo("Administrator created. Log in at /login.")

    @app.cli.command("seed-demo")
    def seed_demo():
        """Sample data. The demo drivers all use the password 'demo12345'."""
        conn = db.get_db()
        if conn.execute("SELECT 1 FROM users WHERE email = 'demo.driver1@example.com'").fetchone():
            raise click.ClickException("Demo data is already there.")
        now = utils.now_str()
        drivers = [
            ("Demo Driver One", "demo.driver1@example.com", "09171234567", "ABC1234", "car"),
            ("Demo Driver Two", "demo.driver2@example.com", "09181234567", "XYZ789", "motorcycle"),
        ]
        ids = []
        for name, email, phone, plate, vtype in drivers:
            cur = conn.execute(
                "INSERT INTO users(full_name, email, phone, password_hash, role, created_at) "
                "VALUES (?, ?, ?, ?, 'user', ?)",
                (name, email, phone, generate_password_hash("demo12345"), now))
            vcur = conn.execute(
                "INSERT INTO vehicles(user_id, plate, vehicle_type, verified, created_at) VALUES (?, ?, ?, 1, ?)",
                (cur.lastrowid, plate, vtype, now))
            ids.append((cur.lastrowid, vcur.lastrowid, plate))
        random.seed(7)
        start = datetime.strptime(now, utils.FMT)
        for i in range(24):
            uid, vid, plate = random.choice(ids + [(None, None, None)])
            when = start - timedelta(days=random.randint(0, 6), hours=random.randint(0, 8),
                                     minutes=random.randint(0, 59))
            vtype = random.choice(["speeding", "honking"])
            stamp = when.strftime(utils.FMT)
            cur = conn.execute(
                "INSERT INTO violations(type, plate, vehicle_id, user_id, speed_kmh, speed_limit, db_level, "
                "detected_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (vtype, plate or "QRS" + str(100 + i), vid, uid,
                 round(random.uniform(12, 26), 1) if vtype == "speeding" else None,
                 10 if vtype == "speeding" else None,
                 round(random.uniform(86, 102), 1) if vtype == "honking" else None, stamp, now))
            conn.execute("UPDATE violations SET record_id = ? WHERE id = ?",
                         (f"RS-{stamp[:10].replace('-', '')}-{cur.lastrowid:05d}", cur.lastrowid))
        conn.commit()
        for uid, _, _ in ids:
            db.recompute_offenses(uid)
        click.echo("Demo data added.")
