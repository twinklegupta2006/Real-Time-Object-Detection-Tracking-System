# Real-Time Object Detection, Tracking & Counting

A real-time computer vision pipeline for detecting, tracking, and counting objects in video. It uses YOLOv8 for object detection and ByteTrack for maintaining persistent object IDs, with line-crossing logic for IN/OUT counting. A lightweight MobileNet-SSD backend is also available as an alternative to YOLOv8.

## Features

- YOLOv8 object detection (MobileNet-SSD also supported)
- ByteTrack multi-object tracking with persistent IDs
- Line-crossing IN/OUT counting — configurable line position, direction, and class filter
- Works with a webcam or a video file
- Annotated video output, CSV logs, and a run summary
- Streamlit web app for running everything from a browser

## Pipeline

```
Video / Webcam
      ↓
YOLOv8 Detection
      ↓
ByteTrack Tracking
      ↓
Persistent IDs
      ↓
Line-Crossing Counting
      ↓
Annotated Output
```

## Tech stack

Python, OpenCV, Ultralytics YOLOv8, ByteTrack, Streamlit

## Installation

```bash
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

YOLOv8 weights download automatically on first run. For MobileNet-SSD, place `deploy.prototxt.txt` and `mobilenet_iter_73000.caffemodel` in `models/`.

## Usage

**CLI**

```bash
python main.py --model yolov8 --track --count --source video.mp4
```

**Streamlit app**

```bash
streamlit run streamlit_app.py
```

## Project structure

```
main.py               CLI entry point
streamlit_app.py       Web UI
src/
  config.py             settings & CLI args
  pipeline.py            core detect -> track -> count -> output loop
  app_core.py             Streamlit's backend logic
  detectors/               MobileNet-SSD & YOLOv8
  tracking/                 ByteTrack wrapper
  counting/                  line-crossing counter
models/                weights go here
```

## Author

Twinkle Gupta
