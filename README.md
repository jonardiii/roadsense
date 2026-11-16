# RoadSense

Automated traffic violation system for a road gate: a Python detector watches the camera, and a
Flask web app stores violations, notifies drivers, and gives admins a review queue.

## Layout

```
detector/              camera side
├── run_detector.py    entry point (camera or video file)
├── detector/          library: pipeline, speed, audio, plates/
├── data_prep/         dataset preparation (run: python -m data_prep.<script>)
├── training/          YOLO training scripts
├── samples/           input photos for manual testing
├── results/           output from plate_test.py
├── models/            trained weights
├── kaggle/            raw Kaggle plate dataset
├── tests/             unittest suite
└── README.md          setup, calibration, training details

web/                   Flask admin and driver portal
├── roadsense/         application package (blueprints, db, templates, static)
├── tests/             unittest suite
└── README.md          setup and API details
```

## Run

```
cd detector
python run_detector.py --config detector_config.json --preview
```

```
cd web
python run.py
```

Open http://127.0.0.1:5000. See each component README for setup, API keys and calibration.

## Tests

```
cd web && python -m unittest discover -s tests -v
cd detector && python -m unittest discover -s tests -v
```

## Where new files go

- Detector library code: `detector/detector/` (plate OCR in `detector/detector/plates/`).
- One-off dataset and training scripts: `detector/data_prep/` and `detector/training/`.
- Test photos: `detector/samples/`. Generated output: `detector/results/`.
- Web routes: the matching module in `web/roadsense/`. Templates: `web/roadsense/templates/<area>/`.
