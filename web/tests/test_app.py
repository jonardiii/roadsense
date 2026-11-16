"""End-to-end checks. Run from the web folder:  python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from werkzeug.security import generate_password_hash  # noqa: E402

from roadsense import create_app, db, utils  # noqa: E402

# Smallest valid JPEG header bytes are enough: the server checks the file signature.
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


class AppTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app({
            "TESTING": True,
            "DATABASE": os.path.join(self.tmp.name, "t.db"),
            "EVIDENCE_DIR": self.tmp.name,
        })
        with self.app.app_context():
            conn = db.get_db()
            conn.execute(
                "INSERT INTO users(full_name,email,password_hash,role,created_at) VALUES (?,?,?,?,?)",
                ("Office Admin", "admin@example.com", generate_password_hash("adminpass1"), "admin", utils.now_str()))
            conn.commit()
            self.key = db.get_setting("api_key")

    def tearDown(self):
        self.tmp.cleanup()

    # ---- helpers ----
    def token(self, client):
        # Pages that render a form create the token. Logged-in clients are redirected away
        # from /login, so set one directly in the session when it's missing.
        with client.session_transaction() as s:
            s.setdefault("_csrf", "test-csrf-token")
            return s["_csrf"]

    def post(self, client, url, data=None, **kw):
        data = dict(data or {})
        data["csrf_token"] = self.token(client)
        return client.post(url, data=data, **kw)

    def login(self, client, email, password):
        return self.post(client, "/login", {"email": email, "password": password})

    def signup(self, client, **over):
        data = dict(full_name="Juan Dela Cruz", email="juan@example.com", phone="0917 123 4567",
                    plate="ABC 1234", vehicle_type="car", password="password1", confirm="password1")
        data.update(over)
        return self.post(client, "/signup", data)

    def api(self, client, **fields):
        fields = {k: v for k, v in fields.items() if v is not None}
        image = fields.pop("image", None)
        video = fields.pop("video", None)
        audio = fields.pop("audio", None)
        if audio:
            import io
            fields["audio"] = (io.BytesIO(audio), "x.wav")
        if video:
            import io
            fields["video"] = (io.BytesIO(video), "x.mp4")
        if image:
            import io
            fields["image"] = (io.BytesIO(image), "x.jpg")
        return client.post("/api/v1/violations", data=fields,
                           headers={"X-API-Key": self.key}, content_type="multipart/form-data")

    # ---- tests ----
    def test_api_needs_key(self):
        c = self.app.test_client()
        self.assertEqual(c.get("/api/v1/health").status_code, 401)
        self.assertEqual(c.get("/api/v1/health", headers={"X-API-Key": "nope"}).status_code, 401)
        self.assertEqual(c.get("/api/v1/health", headers={"X-API-Key": self.key}).status_code, 200)
        cfg = c.get("/api/v1/config", headers={"X-API-Key": self.key}).get_json()
        self.assertEqual(cfg["speed_limit_kmh"], 10.0)

    def test_signup_validation_and_login(self):
        c = self.app.test_client()
        self.assertEqual(self.signup(c, phone="12345").status_code, 400)
        self.assertEqual(self.signup(c, confirm="different").status_code, 400)
        self.assertEqual(self.signup(c, password="short", confirm="short").status_code, 400)
        self.assertEqual(self.signup(c).status_code, 302)
        # duplicate email / plate refused
        c2 = self.app.test_client()
        self.assertEqual(self.signup(c2).status_code, 400)
        # wrong password
        c3 = self.app.test_client()
        self.assertEqual(self.login(c3, "juan@example.com", "wrongwrong").status_code, 401)
        self.assertEqual(self.login(c3, "juan@example.com", "password1").status_code, 302)
        self.assertEqual(c3.get("/me/").status_code, 200)

    def test_roles_and_anonymous_access(self):
        anon = self.app.test_client()
        for url in ("/admin/", "/me/", "/admin/settings", "/admin/reports.csv"):
            r = anon.get(url)
            self.assertEqual(r.status_code, 302, url)
            self.assertIn("/login", r.headers["Location"])
        driver = self.app.test_client()
        self.signup(driver)
        for url in ("/admin/", "/admin/settings", "/admin/vehicles", "/admin/reports.csv"):
            self.assertEqual(driver.get(url).status_code, 403, url)
        admin = self.app.test_client()
        self.login(admin, "admin@example.com", "adminpass1")
        self.assertEqual(admin.get("/admin/").status_code, 200)
        self.assertEqual(admin.get("/me/").status_code, 302)  # admins go to their own dashboard

    def test_horn_can_be_dismissed_without_an_appeal(self):
        admin = self.app.test_client()
        self.login(admin, "admin@example.com", "adminpass1")
        r = self.api(admin, type="honking", plate="HHH1111", db_level="95", detected_at="2026-11-10T09:00:00",
                     sound_note="Vehicle horn, car horn, honking 0.82; Engine 0.30")
        vid = r.get_json()["id"]
        self.assertIn(b"Sound heard", admin.get(f"/admin/violations/{vid}").data)
        self.post(admin, f"/admin/violations/{vid}/decision", {"decision": "dismissed"})
        with self.app.app_context():
            row = db.get_db().execute("SELECT status, offense_no FROM violations WHERE id=?", (vid,)).fetchone()
        self.assertEqual((row["status"], row["offense_no"]), ("dismissed", None))

    def test_unregistered_plate_is_counted_by_plate(self):
        admin = self.app.test_client()
        self.login(admin, "admin@example.com", "adminpass1")
        a = self.api(admin, type="speeding", plate="QQQ1111", speed_kmh="30", detected_at="2026-11-10T08:00:00")
        b = self.api(admin, type="honking", plate="QQQ1111", db_level="95", detected_at="2026-11-10T09:00:00")
        c = self.api(admin, type="speeding", speed_kmh="30", detected_at="2026-11-10T10:00:00")  # plate not read
        self.assertEqual(a.get_json()["offense_no"], 1)
        self.assertEqual(b.get_json()["offense_no"], 2)
        self.assertEqual(c.get_json()["offense_no"], 1)   # plate not read: still counted, on its own
        page = admin.get(f"/admin/violations/{c.get_json()['id']}").data
        self.assertIn(b"Identify the vehicle", page)
        # repeat list shows the plate with 2 violations, and the unidentified record
        rep = admin.get("/admin/repeat").data
        self.assertIn(b"QQQ 1111", rep) if b"QQQ 1111" in rep else self.assertIn(b"QQQ1111", rep)
        self.assertIn(c.get_json()["record_id"].encode(), rep)
        # the administrator reads the plate from the photo: it now counts under QQQ1111 as #3
        r = self.post(admin, f"/admin/violations/{c.get_json()['id']}/set-plate", {"plate": "qqq 1111"})
        self.assertEqual(r.status_code, 302)
        with self.app.app_context():
            n = db.get_db().execute("SELECT offense_no, plate FROM violations WHERE id=?", (c.get_json()["id"],)).fetchone()
        self.assertEqual((n["offense_no"], n["plate"]), (3, "QQQ1111"))
        self.assertIn(b"counted by plate", admin.get(f"/admin/violations/{b.get_json()['id']}").data)
        self.post(admin, f"/admin/violations/{a.get_json()['id']}/delete")     # numbers close up
        with self.app.app_context():
            n = db.get_db().execute("SELECT offense_no FROM violations WHERE id=?", (b.get_json()["id"],)).fetchone()[0]
        self.assertEqual(n, 1)
        # the set plate record is #2 now (b stays 1, c follows)
        with self.app.app_context():
            n = db.get_db().execute("SELECT offense_no FROM violations WHERE id=?", (c.get_json()["id"],)).fetchone()[0]
        self.assertEqual(n, 2)

    def test_video_clip_is_stored_shown_and_deleted(self):
        admin = self.app.test_client()
        self.login(admin, "admin@example.com", "adminpass1")
        mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200
        r = self.api(admin, type="speeding", plate="ABC1234", speed_kmh="30", image=JPEG, video=mp4,
                     detected_at="2026-11-10T10:00:00")
        vid = r.get_json()["id"]
        bad = self.api(admin, type="speeding", plate="XYZ9999", speed_kmh="30", image=JPEG,
                       video=b"not a video at all, just text", detected_at="2026-11-10T11:00:00")
        self.assertEqual(bad.status_code, 201)
        bad_id = bad.get_json()["id"]
        with self.app.app_context():
            self.assertTrue(db.get_db().execute("SELECT video_path FROM violations WHERE id=?", (vid,)).fetchone()[0])
            self.assertIsNone(db.get_db().execute("SELECT video_path FROM violations WHERE id=?", (bad_id,)).fetchone()[0])
        self.assertEqual(admin.get(f"/admin/clip/{vid}").status_code, 200)
        self.assertEqual(admin.get(f"/admin/clip/{bad_id}").status_code, 404)
        page = admin.get(f"/admin/violations/{vid}").data
        self.assertIn(b"<video", page)
        self.assertIn(b'data-zoom="/admin/evidence/', page)   # photo can be zoomed
        self.assertIn(b'id="zv"', page)
        self.assertEqual(len([f for f in os.listdir(self.tmp.name) if f.endswith(".mp4")]), 1)
        self.post(admin, f"/admin/violations/{vid}/delete")
        self.assertEqual([f for f in os.listdir(self.tmp.name) if f.endswith(".mp4")], [])

    def test_sound_clip_is_stored_shown_and_deleted(self):
        admin = self.app.test_client()
        self.login(admin, "admin@example.com", "adminpass1")
        wav = b"RIFF" + b"\x00" * 4 + b"WAVEfmt " + b"\x00" * 60
        r = self.api(admin, type="honking", plate="ABC1234", db_level="95", image=JPEG, audio=wav,
                     detected_at="2026-11-10T10:00:00")
        vid = r.get_json()["id"]
        bad = self.api(admin, type="speeding", plate="XYZ9999", speed_kmh="30", image=JPEG,
                       audio=b"definitely not a sound file, only text " * 3, detected_at="2026-11-10T11:00:00")
        bad_id = bad.get_json()["id"]
        self.assertEqual(admin.get(f"/admin/sound/{vid}").status_code, 200)
        self.assertEqual(admin.get(f"/admin/sound/{bad_id}").status_code, 404)
        self.assertIn(b"<audio", admin.get(f"/admin/violations/{vid}").data)
        self.assertNotIn(b"<audio", admin.get(f"/admin/violations/{bad_id}").data)
        self.assertEqual(len([f for f in os.listdir(self.tmp.name) if f.endswith(".wav")]), 1)
        self.post(admin, f"/admin/violations/{vid}/delete")
        self.assertEqual([f for f in os.listdir(self.tmp.name) if f.endswith(".wav")], [])

    def test_admin_can_delete_violations(self):
        admin = self.app.test_client()
        self.login(admin, "admin@example.com", "adminpass1")
        driver = self.app.test_client()
        self.signup(driver)
        for n, plate in enumerate(("ABC1234", "ABC1234", "XYZ9999")):
            self.api(admin, type="speeding", plate=plate, speed_kmh="30", speed_limit="10",
                     detected_at=f"2026-11-10T10:{n * 10:02d}:00", image=JPEG,
                     record_id=os.urandom(4).hex())
        with self.app.app_context():
            ids = [r["id"] for r in db.get_db().execute("SELECT id FROM violations ORDER BY id")]
        self.assertEqual(len(ids), 3)
        page = admin.get("/admin/violations?type=speeding")
        self.assertIn(b"Type DELETE to confirm", page.data)
        self.assertIn(b"Delete violation", admin.get(f"/admin/violations/{ids[0]}").data)
        files_before = [f for f in os.listdir(self.tmp.name) if f.endswith(".jpg")]
        r = self.post(admin, f"/admin/violations/{ids[0]}/delete")
        self.assertEqual(r.status_code, 302)
        with self.app.app_context():
            self.assertEqual(db.get_db().execute("SELECT COUNT(*) FROM violations").fetchone()[0], 2)
        self.assertLess(len([f for f in os.listdir(self.tmp.name) if f.endswith(".jpg")]), len(files_before))
        # no confirmation word: nothing happens
        self.post(admin, "/admin/violations/delete-matching", {"confirm": "yes"})
        with self.app.app_context():
            self.assertEqual(db.get_db().execute("SELECT COUNT(*) FROM violations").fetchone()[0], 2)
        # only the ones matching the filter
        self.post(admin, "/admin/violations/delete-matching", {"confirm": "DELETE", "q": "XYZ9999"})
        with self.app.app_context():
            left = [r["plate"] for r in db.get_db().execute("SELECT plate FROM violations")]
        self.assertEqual(left, ["ABC1234"])
        self.post(admin, "/admin/violations/delete-matching", {"confirm": "DELETE"})
        with self.app.app_context():
            self.assertEqual(db.get_db().execute("SELECT COUNT(*) FROM violations").fetchone()[0], 0)
        # drivers cannot delete
        self.assertEqual(self.post(driver, "/admin/violations/delete-matching", {"confirm": "DELETE"}).status_code, 403)

    def test_csrf_required(self):
        c = self.app.test_client()
        self.assertEqual(c.post("/login", data={"email": "a", "password": "b"}).status_code, 400)

    def test_login_lockout(self):
        c = self.app.test_client()
        for _ in range(5):
            self.assertEqual(self.login(c, "admin@example.com", "bad-password").status_code, 401)
        self.assertEqual(self.login(c, "admin@example.com", "adminpass1").status_code, 429)

    def test_ingest_validation(self):
        c = self.app.test_client()
        self.assertEqual(self.api(c, type="flying").status_code, 400)
        self.assertEqual(self.api(c, type="speeding").status_code, 400)  # needs speed
        self.assertEqual(self.api(c, type="speeding", speed_kmh="abc").status_code, 400)
        self.assertEqual(self.api(c, type="speeding", speed_kmh="500").status_code, 400)
        self.assertEqual(self.api(c, type="speeding", speed_kmh="15", detected_at="yesterday").status_code, 400)
        self.assertEqual(self.api(c, type="speeding", speed_kmh="15", image=b"not an image").status_code, 400)
        ok = self.api(c, type="speeding", speed_kmh="15.5", plate="xyz 999", image=JPEG,
                      detected_at="2026-11-05T10:00:00+08:00")
        self.assertEqual(ok.status_code, 201)
        body = ok.get_json()
        self.assertFalse(body["registered"])
        self.assertEqual(body["sms"], "not_registered")
        self.assertTrue(body["record_id"].startswith("RS-20261105-"))

    def test_full_flow_verify_offenses_sms_appeal(self):
        driver = self.app.test_client()
        self.signup(driver)
        api = self.app.test_client()

        # Plate is unverified, so the first violation is not linked to the driver.
        r = self.api(api, type="speeding", speed_kmh="14", plate="ABC1234", image=JPEG,
                     detected_at="2026-11-05T09:00:00+08:00")
        self.assertFalse(r.get_json()["registered"])
        first_id = r.get_json()["id"]

        admin = self.app.test_client()
        self.login(admin, "admin@example.com", "adminpass1")
        with self.app.app_context():
            vid = db.get_db().execute("SELECT id FROM vehicles WHERE plate='ABC1234'").fetchone()["id"]
        self.assertEqual(self.post(admin, f"/admin/vehicles/{vid}/verify").status_code, 302)

        # Verification links the earlier violation and numbers it offense 1.
        with self.app.app_context():
            row = db.get_db().execute("SELECT user_id, offense_no FROM violations WHERE id=?", (first_id,)).fetchone()
            self.assertIsNotNone(row["user_id"])
            self.assertEqual(row["offense_no"], 1)

        # Second violation: linked, offense 2, SMS written to the log (default mode is log).
        r = self.api(api, type="honking", db_level="92", plate="ABC 1234", image=JPEG,
                     detected_at="2026-11-05T09:10:00+08:00")
        body = r.get_json()
        self.assertTrue(body["registered"])
        self.assertEqual(body["offense_no"], 2)
        self.assertEqual(body["sms"], "awaiting_review")     # a horn is held until an administrator confirms it
        second_id = body["id"]
        with self.app.app_context():
            self.assertEqual(db.get_db().execute("SELECT COUNT(*) FROM sms_log").fetchone()[0], 0)
        self.assertIn(b"Confirm horn", admin.get(f"/admin/violations/{second_id}").data)
        self.post(admin, f"/admin/violations/{second_id}/confirm-horn")
        with self.app.app_context():
            log = db.get_db().execute("SELECT phone, message FROM sms_log").fetchall()
            self.assertEqual(len(log), 1)
            self.assertEqual(log[0]["phone"], "09171234567")
            self.assertIn("Offense #2", log[0]["message"])

        # Same event repeated within 30 seconds is not stored twice.
        dup = self.api(api, type="honking", db_level="92", plate="ABC1234",
                       detected_at="2026-11-05T09:10:10+08:00")
        self.assertTrue(dup.get_json()["duplicate"])

        # Driver sees own violations and photo; admin sees them too.
        self.assertEqual(driver.get(f"/me/violations/{second_id}").status_code, 200)
        self.assertEqual(driver.get(f"/me/evidence/{second_id}").status_code, 200)
        self.assertEqual(admin.get(f"/admin/evidence/{second_id}").status_code, 200)

        # Another driver cannot open it.
        other = self.app.test_client()
        self.signup(other, email="maria@example.com", plate="")
        self.assertEqual(other.get(f"/me/violations/{second_id}").status_code, 404)
        self.assertEqual(other.get(f"/me/evidence/{second_id}").status_code, 404)

        # Admin cannot close a violation that has no appeal.
        r = self.post(admin, f"/admin/violations/{second_id}/decision", {"decision": "dismissed"})
        self.assertEqual(r.status_code, 302)
        with self.app.app_context():
            self.assertEqual(db.get_db().execute("SELECT status FROM violations WHERE id=?", (second_id,)).fetchone()[0], "flagged")

        # Driver appeals; the other driver can't appeal someone else's record.
        self.assertEqual(self.post(other, f"/me/violations/{second_id}/appeal",
                                   {"kind": "dispute", "message": "not mine at all really"}).status_code, 404)
        self.post(driver, f"/me/violations/{second_id}/appeal", {"kind": "dispute", "message": "I was not honking, it was a bus."})
        with self.app.app_context():
            self.assertEqual(db.get_db().execute("SELECT status FROM violations WHERE id=?", (second_id,)).fetchone()[0], "under_review")

        # Admin dismisses; the driver's offense count drops.
        self.post(admin, f"/admin/violations/{second_id}/decision", {"decision": "dismissed", "note": "Bus horn"})
        with self.app.app_context():
            conn = db.get_db()
            self.assertEqual(conn.execute("SELECT status, offense_no FROM violations WHERE id=?", (second_id,)).fetchone()[1], None)
            self.assertEqual(conn.execute("SELECT status FROM appeals WHERE violation_id=?", (second_id,)).fetchone()[0], "reviewed")

        # Pages render for both roles.
        for url in ("/admin/", "/admin/violations", f"/admin/violations/{second_id}", "/admin/live",
                    "/admin/reports", "/admin/users", "/admin/vehicles", "/admin/settings",
                    "/admin/violations?type=honking&status=resolved&q=ABC&page=2"):
            self.assertEqual(admin.get(url).status_code, 200, url)
        for url in ("/me/", "/me/violations", "/me/vehicles", "/me/profile", f"/me/violations/{first_id}"):
            self.assertEqual(driver.get(url).status_code, 200, url)
        csv = admin.get("/admin/reports.csv?date_from=2026-11-01&date_to=2026-11-30")
        self.assertEqual(csv.status_code, 200)
        self.assertIn("ABC1234", csv.get_data(as_text=True))

    def test_vehicle_kind_is_stored_and_shown(self):
        c = self.app.test_client()
        r = self.api(c, type="speeding", speed_kmh="22", vehicle_class="Motorcycle", detected_at="2026-11-05T10:00:00+08:00")
        self.assertEqual(r.status_code, 201)
        r = self.api(c, type="speeding", speed_kmh="23", vehicle_class="person", detected_at="2026-11-05T11:00:00+08:00")
        self.assertEqual(r.status_code, 201)           # an unknown kind is ignored, not an error
        with self.app.app_context():
            rows = db.get_db().execute("SELECT vehicle_class FROM violations ORDER BY id").fetchall()
        self.assertEqual([r["vehicle_class"] for r in rows], ["motorcycle", None])

    def test_plates_are_shown_with_a_space_for_cars_only(self):
        from roadsense.utils import format_plate
        self.assertEqual(format_plate("JAC7183"), "JAC 7183")
        self.assertEqual(format_plate("ABC123"), "ABC 123")
        self.assertEqual(format_plate("856KCY"), "856KCY")        # motorcycle and tricycle plates
        self.assertEqual(format_plate("1234AB"), "1234AB")
        self.assertEqual(format_plate("AB12345"), "AB12345")
        self.assertEqual(format_plate("1201401686"), "1201-401686")
        self.assertEqual(format_plate(None), "")

    # ---- SMS delivery and retry ----
    def live_driver(self):
        """A verified vehicle with an owner who has a phone, SMS in live mode with a key."""
        with self.app.app_context():
            conn = db.get_db()
            cur = conn.execute("INSERT INTO users(full_name,email,phone,password_hash,role,created_at) "
                               "VALUES ('Dana','d@example.com','09171234567','x','user',?)", (utils.now_str(),))
            conn.execute("INSERT INTO vehicles(user_id,plate,vehicle_type,verified,created_at) "
                         "VALUES (?, 'ABC1234', 'car', 1, ?)", (cur.lastrowid, utils.now_str()))
            conn.commit()
            db.set_setting("sms_mode", "live")
            db.set_setting("semaphore_api_key", "key")

    def fake_gateway(self, outcome):
        """Replace the network call: outcome is 'ok', 'offline', 'busy' or 'rejected'."""
        from urllib.error import HTTPError, URLError
        from roadsense import sms

        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b"[]"

        def fake(req, timeout=10):
            self.calls = getattr(self, "calls", 0) + 1
            if outcome == "offline":
                raise URLError("no internet")
            if outcome == "busy":
                raise HTTPError("u", 503, "busy", {}, None)
            if outcome == "rejected":
                raise HTTPError("u", 401, "bad key", {}, None)
            return Resp()
        sms.urlopen = fake

    def sms_state(self):
        with self.app.app_context():
            row = db.get_db().execute("SELECT status, attempts FROM sms_log").fetchone()
            vstat = db.get_db().execute("SELECT sms_status FROM violations").fetchone()[0]
            return row["status"], row["attempts"], vstat

    def test_sms_is_queued_when_offline_and_sent_when_back(self):
        from roadsense import sms
        original = sms.urlopen
        self.addCleanup(setattr, sms, "urlopen", original)
        self.live_driver()
        self.fake_gateway("offline")
        c = self.app.test_client()
        r = self.api(c, type="speeding", speed_kmh="22", plate="ABC1234", detected_at="2026-11-05T10:00:00+08:00")
        self.assertEqual(r.get_json()["sms"], "queued")
        self.assertEqual(self.sms_state(), ("queued", 1, "queued"))
        with self.app.app_context():
            self.assertEqual(sms.retry_pending(), 0)          # still offline
        self.assertEqual(self.sms_state()[0], "queued")
        self.assertEqual(self.sms_state()[1], 2)
        self.fake_gateway("ok")
        with self.app.app_context():
            self.assertEqual(sms.retry_pending(), 1)
        self.assertEqual(self.sms_state(), ("sent", 3, "sent"))

    def test_a_dead_connection_costs_one_attempt_not_one_per_message(self):
        from roadsense import sms
        original = sms.urlopen
        self.addCleanup(setattr, sms, "urlopen", original)
        self.live_driver()
        self.fake_gateway("offline")
        c = self.app.test_client()
        for i in range(3):
            self.api(c, type="speeding", speed_kmh="22", plate="ABC1234",
                     detected_at=f"2026-11-05T1{i}:00:00+08:00")
        self.calls = 0
        with self.app.app_context():
            sms.retry_pending()
        self.assertEqual(self.calls, 1)

    def test_a_rejected_text_is_not_retried(self):
        from roadsense import sms
        original = sms.urlopen
        self.addCleanup(setattr, sms, "urlopen", original)
        self.live_driver()
        self.fake_gateway("rejected")
        r = self.api(self.app.test_client(), type="speeding", speed_kmh="22", plate="ABC1234",
                     detected_at="2026-11-05T10:00:00+08:00")
        self.assertEqual(r.get_json()["sms"], "failed")
        self.calls = 0
        with self.app.app_context():
            sms.retry_pending()
        self.assertEqual(self.calls, 0)

    def test_gateway_trouble_is_queued_and_gives_up_eventually(self):
        from roadsense import sms
        original = sms.urlopen
        self.addCleanup(setattr, sms, "urlopen", original)
        self.live_driver()
        self.fake_gateway("busy")
        r = self.api(self.app.test_client(), type="speeding", speed_kmh="22", plate="ABC1234",
                     detected_at="2026-11-05T10:00:00+08:00")
        self.assertEqual(r.get_json()["sms"], "queued")
        with self.app.app_context():
            db.get_db().execute("UPDATE sms_log SET attempts = ?", (sms.MAX_ATTEMPTS - 1,))
            db.get_db().commit()
            sms.retry_pending()
        self.assertEqual(self.sms_state()[0], "failed")
        self.assertIn("Gave up", self.detail())

    def detail(self):
        with self.app.app_context():
            return db.get_db().execute("SELECT detail FROM sms_log").fetchone()[0]

    def test_admin_can_resend_a_queued_text(self):
        from roadsense import sms
        original = sms.urlopen
        self.addCleanup(setattr, sms, "urlopen", original)
        self.live_driver()
        self.fake_gateway("offline")
        self.api(self.app.test_client(), type="speeding", speed_kmh="22", plate="ABC1234",
                 detected_at="2026-11-05T10:00:00+08:00")
        self.fake_gateway("ok")
        c = self.app.test_client()
        self.login(c, "admin@example.com", "adminpass1")
        page = c.get("/admin/violations/1").get_data(as_text=True)
        self.assertIn("Send the text again now", page)
        self.post(c, "/admin/violations/1/resend-sms", {})
        self.assertEqual(self.sms_state()[0], "sent")

    def test_settings_validation(self):
        admin = self.app.test_client()
        self.login(admin, "admin@example.com", "adminpass1")
        base = dict(speed_limit="10", honk_db="85", honk_min_seconds="1", sms_mode="log",
                    semaphore_sender="", stream_url="")
        self.post(admin, "/admin/settings", dict(base, speed_limit="9999"))
        self.post(admin, "/admin/settings", dict(base, stream_url="javascript:alert(1)"))
        with self.app.app_context():
            self.assertEqual(db.get_setting("speed_limit"), "10")
            self.assertEqual(db.get_setting("stream_url"), "")
        self.post(admin, "/admin/settings", dict(base, speed_limit="12", allow_direct_review="on"))
        with self.app.app_context():
            self.assertEqual(db.get_setting("speed_limit"), "12")
            self.assertEqual(db.get_setting("allow_direct_review"), "1")

    def test_escaping(self):
        driver = self.app.test_client()
        self.signup(driver, full_name="<script>alert(1)</script>")
        html = driver.get("/me/").get_data(as_text=True)
        self.assertNotIn("<script>alert(1)</script>", html)


if __name__ == "__main__":
    unittest.main()
