"""Entry point for the real-time object detection pipeline.

Usage:
    python main.py                              # MobileNet-SSD, webcam 0
    python main.py --model yolov8                # YOLOv8n instead
    python main.py --model yolov8 --track         # YOLOv8n + ByteTrack (persistent IDs)
    python main.py --model yolov8 --count         # + line-crossing counting (implies --track)
    python main.py --source video.mp4             # run on a video file instead
    python main.py --no-display --no-video        # headless, CSV/log only
"""
import logging
import sys

from src.config import config_from_args
from src.counting import create_counter
from src.detectors import create_detector
from src.pipeline import ObjectDetectionPipeline
from src.tracking import create_tracker


def main(argv=None) -> None:
    cfg = config_from_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(cfg.log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    cfg.validate_paths()
    detector = create_detector(cfg)
    tracker = create_tracker(cfg) if cfg.tracking_enabled else None
    counter = create_counter(cfg) if cfg.counting_enabled else None
    pipeline = ObjectDetectionPipeline(detector, cfg, tracker=tracker, counter=counter)
    pipeline.run()


if __name__ == "__main__":
    main()
