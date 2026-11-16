"""A tiny live video feed for the web system's "Live camera" page.

The detector keeps the newest annotated picture and serves it as a motion-JPEG stream, which a plain
<img> tag in the browser can show. By default it listens on this laptop only (127.0.0.1), because the
picture shows plates and people.
"""
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

log = logging.getLogger("roadsense.live")


class LiveFeed:
    def __init__(self, port=5001, host="127.0.0.1", width=960, fps=8.0, quality=70):
        self.width, self.min_gap, self.quality = width, 1.0 / max(1.0, fps), quality
        self._jpeg = None
        self._lock = threading.Lock()
        self._last = 0.0
        feed = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path.split("?")[0] not in ("/video_feed", "/"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, private")
                self.end_headers()
                last = None
                try:
                    while True:
                        data = feed.latest()
                        if data is not None and data is not last:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                            last = data
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        log.info("Live feed at http://%s:%d/video_feed", "localhost" if host == "127.0.0.1" else host, port)

    def wanted(self):
        return time.monotonic() - self._last >= self.min_gap

    def update(self, image):
        self._last = time.monotonic()
        h, w = image.shape[:2]
        if w > self.width:
            image = cv2.resize(image, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()

    def latest(self):
        with self._lock:
            return self._jpeg

    def close(self):
        self.server.shutdown()
