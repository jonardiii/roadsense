# RoadSense detector

Watches the camera, measures vehicle speed, reads licence plates, and sends violations (with a photo)
to the web system. Detects horns from a microphone too, and matches them to the vehicle in view.

Put this folder next to the `web` folder:

```
roadsense-web/
  web/
  detector/
```

## 1. Install (Windows, in a new terminal in the `detector` folder)

```
python -m venv .venv
.venv\Scripts\activate
```

**Install PyTorch with GPU support first.** If you skip this, the next command installs a CPU-only
PyTorch on Windows and everything runs slowly. Get the current command for your CUDA version from
pytorch.org (Windows, Pip, CUDA). It looks like this, on one line:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install flask
python -c "import torch; print(torch.cuda.is_available())"
```

The last line must print `True`. (Flask is only needed by the tests that start the real web system.)

**Check the preview window works.** The OCR package installs a second copy of OpenCV without window
support, which can break `calibrate.py` and `--preview`. Test it:

```
python -c "import cv2; cv2.namedWindow('t'); cv2.destroyAllWindows(); print('windows ok')"
```

If that prints an error ("The function is not implemented"), fix it with:

```
pip uninstall -y opencv-python-headless
pip install --force-reinstall --no-deps opencv-python
```

## 2. Check the code (no camera needed)

```
python -m unittest discover -s tests -v
```

The first run downloads the OCR models (about 100 MB), so it takes a minute longer. Every test should pass.

## 3. Mark the speed zone on the road

Mark a rectangle on the road with chalk or cones, about 3 m wide and 6 to 8 m long, with the long
side along the road. Measure both sides with a tape. With the camera mounted where it will stay:

```
python calibrate.py --source "rtsp://USER:PASSWORD@CAMERA_IP:554/stream1" --width 3 --length 6
```

Click the four corners in this order, as the camera sees them: near-left, near-right, far-right,
far-left. Press S to save, then open `calibration_preview.png` and check that the yellow zone sits
on your marks and the white lines are one metre apart on the road. Repeat if the camera moves.

**Keep the whole zone inside the frame**, where vehicles are fully visible. A vehicle cut off by the
edge of the picture has a box that moves slower than the vehicle itself. The detector ignores such
frames, but it can only measure what it sees, so leave margin around the zone.

The Tapo camera's RTSP address and account are set in the Tapo app (Camera settings, Advanced
settings, Camera account). Use `stream2` (lower resolution) if the laptop struggles with `stream1`.

## 4. Plate reading

Plate reading needs a model trained to find plates. Without one, violations are still sent, with
the photo, and the administrator reads the plate by eye.

**Get data.** Your references list two plate datasets:
- Roboflow "Philippine license plates" (University of Southeastern Philippines). Open the dataset
  page, choose Download Dataset, format **YOLOv11**, and unzip it to `data/roboflow_ph`.
- Kaggle "Car License Plate Detection" by andrewmvd. Unzip it so you have an `images` and an
  `annotations` folder, then convert it:
  `python -m data_prep.prepare_dataset voc --images kaggle/images --annotations kaggle/annotations --out data/kaggle`
  (These are not Philippine plates, so they help the model find plates in general, not read ours.)
- **Your own camera, which matters most.** Take the video of passing vehicles, then:
  `python -m data_prep.extract_frames --video clip.mp4 --out data/own_raw --every 5`
  Upload those frames to Roboflow or CVAT, draw a box around every plate, and export as YOLOv11
  to `data/own_frames`. Aim for 200 to 300 labelled frames.

**Combine and train.** Your own frames are repeated 3 times so they count for more:

```
python -m data_prep.prepare_dataset merge --inputs data/kaggle data/roboflow_ph data/own_frames --repeats 1 1 3 --out data/plates
python training/train_plate_model.py --data data/plates/data.yaml
```

Training prints precision, recall and mAP on images it never trained on. Put those numbers in
Chapter 4. Then set `"plate_model": "models/plate.pt"` in `detector_config.json`.

**How a plate is read.** For each vehicle the detector finds the plate on every second frame, skips
tiny or blurry views, and reads the characters with OCR. It fixes letter and digit mix-ups using
the Philippine plate formats (for example `ABC1Z34` becomes `ABC1234`), then votes character by
character across all frames of that vehicle. A plate that matches a registered, verified plate
(fetched from the web system) gets a small boost, and a one-character difference is corrected to the
registered plate only if some frame actually saw that character.

The result has a confidence from 0 to 1. At or above `min_plate_confidence` (default 0.6) the plate
is sent and the web system links the violation to the driver and sends the SMS. Below it, the
violation is sent **without** a plate, and the photo shows `plate unclear: ABC1234?` so the
administrator can read it. This is deliberate: a wrong plate would send an SMS to the wrong person.

### If you cannot download the datasets

You do not need them to get started. Train on your own labelled frames only: label 300 to 500 frames
from your camera with a free tool (CVAT at cvat.ai, or LabelImg, which runs on your laptop and needs no
account), export as YOLO, and run:

```
python -m data_prep.prepare_dataset merge --inputs data/own_frames --repeats 1 --out data/plates
python training/train_plate_model.py --data data/plates/data.yaml
```

The model starts from pretrained YOLO11 weights, so a few hundred frames of your own camera is enough to
find plates on your road. More data from other sources only helps it find plates in unusual conditions.

## Live camera on the website

While the detector runs it serves the annotated video at `http://localhost:5001/video_feed` (set
`live_port` in `detector_config.json`, 0 turns it off). In the web system, open Settings and put that
address in **Live feed address**, then open **Live camera**. It listens on this laptop only. Set
`"live_host": "0.0.0.0"` to let other devices on your network watch, but the picture shows plates and
people, so do that only on a network you trust.

## 6. Horn detection

Plug in the microphone first, then test it on its own (no camera needed):

```
python horn_tool.py devices
python horn_tool.py meter
```

Honk, talk, clap and drive past. The meter shows the level, how far above the background it is, and how
tone-like it is, and prints `>>> HORN` when it decides. To turn it on in the detector, set in
`detector_config.json`:

```
"horn_enabled": true,
"audio_device": "",
"video_lag_s": 1.0
```

**How a horn is decided.** A frame counts as horn-like if it is at least `horn_min_excess_db` (10 dB)
louder than the rolling background, most of its energy sits in a few narrow peaks, and it stays the same
sound for `horn_min_duration_s` (0.2 s). Speech, engines, wind and passing traffic fail one of these.
Beeps within `horn_cooldown_s` (2 s) are one event.

**Which vehicle honked.** One microphone cannot tell direction. The detector looks at the vehicles in
view at that moment. If exactly one is in view, its plate (if read) is attached. If several are in view,
the record is sent **without a plate**, with all of them marked on the photo, for the administrator to
decide. If none is in view, nothing is sent. Vehicles narrower than `horn_min_box_px` (80 px) are too far
away to count.

**Set `video_lag_s`.** The camera picture runs behind the sound, usually 0.5 to 2 seconds for an RTSP
stream. Stand next to the camera, clap once, and watch the preview: the seconds between the clap and the
picture of it is your value.

**dB level (optional).** The microphone gives a level relative to full scale, not real loudness. To send
an estimated dB value, play a steady tone near the microphone, read a phone sound-meter app next to it,
note what the `meter` shows, and set `mic_offset_db` to the difference (app dB minus meter dB). Treat the
result as an estimate; it is not a calibrated measurement.

**Measure it on your own site.** Synthetic tests show the logic works, not how it behaves on your road.
Record sound with `horn_tool.py record`, put honks in one folder and everything else the microphone hears
(traffic, voices, engines, a bell) in another, then:

```
python horn_tool.py evaluate --horn data/horn --other data/other
```

It prints the share of horns found and false alarms per hour. If horns are missed, lower `--min-excess-db`
or `--min-tonal`; if there are false alarms, raise them. Put these numbers in Chapter 4. The HornBase
dataset from your references can supply horn recordings.

**Try it on photos (no camera).** Put phone photos of parked cars (2 to 5 m away, plate visible) in a
folder, naming each after its real plate, for example `ABC1234_01.jpg`:

```
python plate_test.py --images samples/photos --model models/plate.pt
```

It prints what it found and read for each photo, how many it got exactly right, and saves marked-up
copies in `results/plate_test_out`. One photo per vehicle is harder than the real detector's job, which votes over
many frames, so treat the result as a lower bound.

**Two text readers.** The detector can read plate characters with `easyocr` (the default, a general
reader) or `fastplate` (the fast-plate-ocr package, trained only on plate crops, runs on the CPU).
Install the second with `pip install "fast-plate-ocr[onnx]"`, then compare them on the same photos:

```
python distance_test.py --images samples/photos_phone --model models/plate.pt --camera-like --ocr easyocr
python distance_test.py --images samples/photos_phone --model models/plate.pt --camera-like --ocr fastplate
```

Use whichever reads more plates exactly right at the plate widths you expect, and set `"ocr_engine"` in
`detector_config.json`. Note that `fastplate` reports confidence very close to 1 even on poor reads, so
judge it by the exactly-right counts, not by its confidence numbers.

**How small can a plate be?** `python distance_test.py --images samples/photos_phone` shrinks your named photos
so the plate is 160 down to 30 pixels wide and shows where reading stops working. Plan the camera so
plates are at least that wide.

## 5. Configure and run

Copy `detector_config.example.json` to `detector_config.json`, paste the API key from the web
Settings page, and set the camera address. Start the web system first, then:

```
python run_detector.py --config detector_config.json --preview
```

To try a recorded clip instead of the camera, add `--source clip.mp4`. Violations that can't be
delivered (website closed, Wi-Fi down) wait in the `outbox` folder and are sent when the connection
returns.

## What the tests show, and what they don't

**Speed.** A simulation of vehicles at known speeds through a calibrated zone gives an average
error of about 0.2 km/h at 6 to 40 km/h, even with a rough stream (10% bad boxes, 25 ms frame-time
jitter) at campus speeds. The real YOLO11 detector was also run on a video of a photograph sliding
across a calibrated road at 6, 12, 20 and 30 km/h, and measured 5.9, 11.8, 19.7 and 29.6 km/h. That
video has no real-road perspective, shadows or compression, so treat it as a check that the chain
works, not as your accuracy. In real use the error also includes your tape measurement, where you
clicked the corners, and how well the box bottom follows the vehicle on the road.

**Measure the real error before the defense:** drive a vehicle at known speeds (a car with a GPS
speedometer app) through the zone a dozen times and compare. Use that error to set
`speed_margin_kmh`, so vehicles aren't flagged for being within the measurement error of the limit.

**Plates.** The voting logic was tested by simulation: with independent character errors on 5%,
10% and 20% of characters per frame, a single frame reads 72%, 58% and 34% of plates right, while
voting over 12 frames reads essentially all of them, and no wrong plate was sent. Real OCR errors
repeat from frame to frame more than these simulated ones, so real gains will be smaller. The
limit that voting cannot fix is a mistake made on most frames: if 60% of frames misread the same
letter, the vote picks the wrong letter with high confidence. Only the registered-plate check
rescues that case, and only for registered vehicles. The real OCR engine reads a clearly drawn
plate correctly, but the only real measure of plate accuracy is your own labelled test frames:
hold some back, run the detector on them, and count.


## Vehicle types: van, jeepney, tricycle

The standard YOLO model only knows car, motorcycle, bus and truck, so vans and jeepneys are often called "truck".
To teach it the Philippine types, train it on Roboflow datasets that have those classes:

1. Download the datasets from Roboflow Universe in **YOLOv11** (or YOLOv8) format and unzip each into `data\`.
   Two that have the classes: "Vehicle Classification Philippines" (thesis-vdkcz: car, truck, bus, van, jeepney,
   motorcycles, tricycles) and "Vehicle Detection PH" (vision-drive-2: 14 classes including Jeepney, Van, Tricycle).
2. Merge them: `python -m data_prep.prepare_vehicles --inputs data\ds1 data\ds2 --out data\vehicles --max-images 8000`
   It prints how every class name was mapped. Check that list.
3. Train: `python training/train_vehicle_model.py --data data\vehicles\data.yaml --batch 8 --workers 0`
4. In `detector_config.json` set `"model": "models/vehicle.pt"` and `"vehicle_classes": []`.

## Horn placement (one microphone cannot tell direction)

- Put the microphone close to the gate, on the side the vehicles pass. A horn far away reaches it much quieter.
- Calibrate `mic_offset_db` with a sound meter app, then set the horn level threshold on the website's Settings
  page. A horn quieter than that is ignored. Both the level and the shortest horn (seconds) are read from the website.
- `horn_zone_only` (on by default): a horn is blamed only on a vehicle inside the calibrated area.
- On the website, horn records wait for an administrator ("Hold horn records for review"). The driver is texted
  only after the administrator watches the clip with sound and presses Confirm.

## Horn classifier (tells a horn from shouting)

The frequency check alone can be fooled by a loud shout. YAMNet, a sound model pre-trained by Google on AudioSet
(it has "Vehicle horn, car horn, honking", "Shout", "Yell", "Speech"), is used as a second check.

1. Download the YAMNet ONNX file, for example from https://huggingface.co/audiomagic/yamnet-onnx or
   https://huggingface.co/andrelgomes/yamnet-onnx, and save it as `models\yamnet.onnx`.
2. Restart the detector. It prints "Horn sound model on" when it is loaded. Without the file it keeps working with
   the frequency check only.
3. `horn_model_min_score` (default 0.3) is how sure the model must be that the sound is a horn. Lower it if real
   horns are being rejected, raise it if shouts still get through.
4. Test it on your own recordings: `python horn_model_test.py recording.wav` prints the horn and voice scores.

### Measuring horn accuracy (for the paper)

`python horn_eval.py --dataset urbansound8k --root data\UrbanSound8K --csv horn_results.csv` (or `--dataset esc50`)
tests every clip and prints, per kind of sound, how many were accepted as a horn, with and without the sound model,
plus precision and recall and a table of `horn_model_min_score` values. `pip install soundfile` first.

### Fine-tuning the horn detector on horn recordings

`python horn_train_head.py --dataset hornbase --root data\HornBase` trains a small layer on top of YAMNet with the
HornBase train clips, prints the accuracy on its test clips, and saves `models\horn_head.npz`. The detector uses it
automatically. Add `--normalize` to bring quiet clips to one loudness first. To add your own horns from the entrance, make a
folder `my_sounds` with subfolders `horn` and `other` of .wav files and add `--extra data\my_sounds`.
Test it with `python horn_eval.py --dataset hornbase --root data\HornBase --head models\horn_head.npz`
(that test includes the training clips; the accuracy that counts is the one `horn_train_head.py` prints for its test clips).

## Sound clips, camera audio, and testing horns with a video

**A separate sound clip with each violation.** Next to the video clip, every speeding or honking record now
also carries the same 10 seconds of sound as its own file (WAV). The website shows it as an audio player on
the violation page (admin and driver). It comes from the external microphone only, and only when
`horn_enabled` is on and the microphone is working.

**The camera's own microphone is never used.** The program asks the RTSP camera for the picture only
(`"camera_audio_off": true`, the default), so the camera's sound is not even received. All sound in clips and
horn detection comes from the external microphone. For a complete switch-off at the source, also turn the
camera's microphone off in the Tapo app (the camera's settings).

**Testing horns with a video that has sound** (a clip downloaded from YouTube, or a phone recording):

```
python run_detector.py --config detector_config.json --source data\yt\clip.mp4 --preview
```

When the file has a sound track, its sound is fed to the horn detector in step with the picture, so a honk is
matched to the vehicles in that very picture (no `video_lag_s` shift is needed: it is 0 for files). The video
clip and sound clip of each violation use the file's sound too. Needs `imageio-ffmpeg`. Live streams
(YouTube live, RTSP) are video only; their sound is not used.

## Multicabs (and other types the model confuses)

The vehicle types are: car, van, jeepney, tricycle, motorcycle, bus, truck, **multicab**. Pickups
(Hilux, Strada, Navara...) are merged into "car", and multicabs have their own type, so they are no
longer taken for trucks, vans or jeepneys. Public datasets rarely have multicabs, so for those (and for any
type the model still confuses) use frames of your own camera:

1. Get sharp frames: `python -m data_prep.extract_frames --video clip.mp4 --out data\own_frames --every 15`
2. Upload them to Roboflow, draw a box on each vehicle and give it its type (use the names above), export
   as "YOLOv8" format and unzip it into `data\own_frames`.
3. Merge it with the public sets, repeating your own frames 3 times so the model pays attention to them:
   `python -m data_prep.prepare_vehicles --inputs data\ph_vehicles1 data\ph_vehicles2 data\own_frames*3 --out data\vehicles`
4. Train again, starting from the model you have (shorter, 30 epochs):
   `python training/train_vehicle_model.py --data data\vehicles\data.yaml --model models\vehicle.pt --epochs 30`

Aim for at least 150 boxes of each type you want the model to know (the merge prints the count per type).
