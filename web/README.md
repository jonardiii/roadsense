# RoadSense web system

Admin and driver portal plus the API your detector program sends violations to.
Built with Flask and SQLite, so the web system and the detector are both Python.

## Run it (Windows)

```
cd web
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

flask --app roadsense create-admin
python run.py
```

Open http://127.0.0.1:5000. The database and secret key are created in `instance/` the first time you run it.

Optional, for screenshots and your demo: `flask --app roadsense seed-demo` adds two sample drivers (password `demo12345`) and 24 sample violations. Don't run it on the database you use for real testing.

Run the tests: `python -m unittest discover -s tests -v`

## How a violation travels

1. The detector posts a violation to `POST /api/v1/violations` with the `X-API-Key` header.
2. If the plate matches a **verified** vehicle, the violation is linked to that driver, numbered as offense 1, 2, 3, and an SMS is queued.
3. If not, it is still saved (with the photo) and shows up as "not matched to a driver". No SMS is sent.
4. The driver sees it under My violations and can send an appeal. The administrator reviews it.

A plate counts only after an administrator verifies it (Vehicles page). Without that step, anyone could register another person's plate and read their violations.

## Detector API

Send the key shown on the admin Settings page in the `X-API-Key` header.

| Call | Purpose |
| --- | --- |
| `GET /api/v1/config` | Speed limit, horn dB threshold and minimum horn length from Settings |
| `GET /api/v1/plates` | Verified plates, for matching OCR text that is off by one character |
| `POST /api/v1/violations` | Report a violation (multipart form with an optional `image`, or JSON) |
| `POST /api/v1/heartbeat` | Tell the dashboard the detector is running (`{"info": "12 fps"}`) |
| `GET /api/v1/health` | Check the key and connection |

Violation fields: `type` (`speeding` or `honking`), `speed_kmh` (required for speeding), `db_level`, `plate`, `plate_confidence` (0 to 1), `detected_at` (ISO 8601, defaults to now), `image` (JPEG or PNG).

The same plate and type within 30 seconds is stored once, so a detector that fires repeatedly won't spam a driver.

```python
import requests
requests.post(
    "http://127.0.0.1:5000/api/v1/violations",
    headers={"X-API-Key": "YOUR_KEY"},
    data={"type": "speeding", "speed_kmh": 14.2, "plate": "ABC1234", "plate_confidence": 0.91},
    files={"image": open("evidence.jpg", "rb")},
    timeout=10,
)
```

## SMS (Semaphore)

Settings has three modes. **Test** writes the message to the SMS log without sending (the default). **Live** sends through Semaphore using the API key you save in Settings, or the `SEMAPHORE_API_KEY` environment variable. Try Test first so you don't spend credits while developing.

## Differences from the paper

- Flask and SQLite replace Laravel and MongoDB. Update Chapter 3 (software requirements, architecture, data flow) to match, or tell me and I'll build the Laravel version.
- Offense steps follow the paper: 1st offense, appeal or letter of apology; 2nd, report to the office; 3rd, gate pass suspension.
- Administrators review only violations that have an appeal, as the paper says. The Settings page has a switch to allow direct review, which you'll want when measuring false detections.

## Before the defense

- Put the site behind HTTPS if it leaves your laptop (for example a reverse proxy), and keep `host="127.0.0.1"` in `run.py` unless other devices need access.
- Back up `instance/roadsense.db` and the `instance/evidence/` folder. Violation records contain personal data.
