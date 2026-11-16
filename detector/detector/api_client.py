"""Talks to the RoadSense web system. Violations that can't be delivered are kept on disk
and retried, so a Wi-Fi drop or a restarted website never loses a record."""
import json
import logging
import os
import shutil
import uuid

import requests

log = logging.getLogger("roadsense.api")

SENT, QUEUED, REJECTED = "sent", "queued", "rejected"


class ApiClient:
    def __init__(self, base_url, api_key, outbox_dir="outbox", timeout=8):
        self.base = base_url.rstrip("/") + "/api/v1"
        self.headers = {"X-API-Key": api_key}
        self.timeout = timeout
        self.outbox = outbox_dir
        self.rejected_dir = os.path.join(outbox_dir, "rejected")
        os.makedirs(self.outbox, exist_ok=True)
        os.makedirs(self.rejected_dir, exist_ok=True)

    def _get(self, path):
        try:
            r = requests.get(self.base + path, headers=self.headers, timeout=self.timeout)
        except requests.RequestException as exc:
            log.warning("Could not reach the web system: %s", exc)
            return None
        if r.status_code == 401:
            log.error("The web system refused the API key. Copy the current key from Settings.")
            return None
        if not r.ok:
            log.warning("GET %s returned HTTP %s", path, r.status_code)
            return None
        return r.json()

    def config(self):
        """Thresholds from the web Settings page, or None if unreachable."""
        return self._get("/config")

    def plates(self):
        data = self._get("/plates")
        return data["plates"] if data else None

    def heartbeat(self, info=""):
        try:
            r = requests.post(self.base + "/heartbeat", json={"info": info},
                              headers=self.headers, timeout=self.timeout)
            return r.ok
        except requests.RequestException:
            return False

    def _post_violation(self, fields, jpeg, video=None, audio=None):
        files = {}
        if jpeg:
            files["image"] = ("evidence.jpg", jpeg, "image/jpeg")
        if video:
            files["video"] = ("evidence.mp4", video, "video/mp4")
        if audio:
            files["audio"] = ("evidence.wav", audio, "audio/wav")
        return requests.post(self.base + "/violations", data=fields, files=files or None,
                             headers=self.headers,
                             timeout=max(self.timeout, 40) if (video or audio) else self.timeout)

    def send_violation(self, fields, jpeg=None, video=None, audio=None):
        """Send now; on a network or server problem keep it in the outbox. Returns a status."""
        try:
            r = self._post_violation(fields, jpeg, video, audio)
        except requests.RequestException as exc:
            log.warning("Web system unreachable (%s). Saved to the outbox.", exc)
            self._enqueue(fields, jpeg, video, audio)
            return QUEUED
        return self._handle(r, fields, jpeg, enqueue_on_retryable=True, video=video, audio=audio)

    def _handle(self, r, fields, jpeg, enqueue_on_retryable, video=None, audio=None):
        if r.status_code in (200, 201):
            return SENT
        if r.status_code == 401 or r.status_code >= 500:
            log.error("Web system answered HTTP %s. Keeping the record to retry.", r.status_code)
            if enqueue_on_retryable:
                self._enqueue(fields, jpeg, video, audio)
            return QUEUED
        log.error("Web system rejected a record (HTTP %s): %s", r.status_code, r.text[:200])
        self._stash(fields, jpeg, self.rejected_dir, video, audio)
        return REJECTED

    def _stash(self, fields, jpeg, folder, video=None, audio=None):
        name = uuid.uuid4().hex
        # photo and clip first, the .json last: the outbox only picks up a record once its json exists
        if jpeg:
            with open(os.path.join(folder, name + ".jpg"), "wb") as fh:
                fh.write(jpeg)
        if video:
            with open(os.path.join(folder, name + ".mp4"), "wb") as fh:
                fh.write(video)
        if audio:
            with open(os.path.join(folder, name + ".wav"), "wb") as fh:
                fh.write(audio)
        with open(os.path.join(folder, name + ".json"), "w", encoding="utf-8") as fh:
            json.dump(fields, fh)
        return name

    def _enqueue(self, fields, jpeg, video=None, audio=None):
        self._stash(fields, jpeg, self.outbox, video, audio)

    def pending(self):
        return sorted(f for f in os.listdir(self.outbox) if f.endswith(".json"))

    def flush_outbox(self):
        """Retry queued records, oldest first. Returns how many were delivered."""
        delivered = 0
        for name in self.pending():
            path = os.path.join(self.outbox, name)
            jpg_path = path[:-5] + ".jpg"
            with open(path, encoding="utf-8") as fh:
                fields = json.load(fh)
            jpeg = video = audio = None
            mp4_path = path[:-5] + ".mp4"
            wav_path = path[:-5] + ".wav"
            if os.path.exists(jpg_path):
                with open(jpg_path, "rb") as fh:
                    jpeg = fh.read()
            if os.path.exists(mp4_path):
                with open(mp4_path, "rb") as fh:
                    video = fh.read()
            if os.path.exists(wav_path):
                with open(wav_path, "rb") as fh:
                    audio = fh.read()
            try:
                r = self._post_violation(fields, jpeg, video, audio)
            except requests.RequestException:
                break  # still offline; try again later
            status = self._handle(r, fields, jpeg, enqueue_on_retryable=False, video=video, audio=audio)
            if status == QUEUED:
                break
            if status == REJECTED:
                # _handle stashed a copy in rejected/; drop the queued one
                pass
            os.remove(path)
            for extra in (jpg_path, mp4_path, wav_path):
                if os.path.exists(extra):
                    os.remove(extra)
            if status == SENT:
                delivered += 1
        return delivered
