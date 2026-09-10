# Real-Time Object Detection & Tracking System

A computer vision project for detecting, tracking, and counting objects in real-time video using YOLOv8 and ByteTrack.

## Overview

The pipeline detects objects in a video or webcam feed, keeps a persistent ID for each object using ByteTrack, and counts objects as they cross a user-defined line (IN/OUT). YOLOv8 is the main detector, with MobileNet-SSD available as a lighter alternative.

## Streamlit Demo

```bash
streamlit run streamlit_app.py
```

Opens a browser-based interface for uploading a video and running detection, tracking, and counting without touching the command line.

## Usage

```bash
python main.py --model yolov8 --track --count --source video.mp4
```

```bash
python main.py --model mobilenet --source video.mp4
```

## Project Structure

```text
main.py              # CLI entry point
streamlit_app.py     # Streamlit application
src/
├── detectors/       # YOLOv8 and MobileNet-SSD detection
├── tracking/        # ByteTrack tracking
├── counting/        # Line-crossing counting
├── pipeline.py      # Main processing pipeline
└── config.py        # Configuration
requirements.txt
```

## Setup

```bash
python -m venv .venv
venv\Scripts\activate       # Windows
# source .venv/bin/activate # macOS/Linux
pip install -r requirements.txt
```

YOLOv8 weights download automatically on first run. MobileNet-SSD weights go in `models/`.

## Tech Stack

Python, OpenCV, YOLOv8, ByteTrack, Streamlit
