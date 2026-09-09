"""Speed benchmark: measures warm-up-excluded inference latency/FPS/detection
counts/confidence distribution for each detector backend, on the same fixed
set of frames, at the same input resolution and confidence threshold, through
the common detector interface (src/detectors).

This measures *speed only* -- see evaluate.py / README for accuracy. A model
being faster here says nothing about which one detects correctly; the two
are reported separately on purpose.

Usage:
    python benchmark.py                          # both models, data/eval/images by default
    python benchmark.py --models yolov8 --device 0
    python benchmark.py --source some_video.mp4 --max-frames 100
"""
import argparse
import logging
import os
import statistics
import time
from typing import List, Optional

import cv2
import numpy as np

from src.config import Config
from src.detectors import create_detector

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def load_frames(source: str, max_frames: int, resize_to: Optional[tuple] = None) -> List[np.ndarray]:
    if os.path.isdir(source):
        files = sorted(
            os.path.join(source, f) for f in os.listdir(source) if f.lower().endswith(IMAGE_EXTENSIONS)
        )[:max_frames]
        frames = [cv2.imread(f) for f in files]
        frames = [f for f in frames if f is not None]
    else:
        video_source = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {source!r}")
        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

    if resize_to:
        frames = [cv2.resize(f, resize_to) for f in frames]
    return frames


def confidence_histogram(confidences: List[float], low: float = 0.5, high: float = 1.0, buckets: int = 5) -> dict:
    if not confidences:
        return {}
    width = (high - low) / buckets
    counts = [0] * buckets
    for c in confidences:
        idx = int((c - low) / width) if c >= low else 0
        idx = min(max(idx, 0), buckets - 1)
        counts[idx] += 1
    labels = [f"{low + i * width:.2f}-{low + (i + 1) * width:.2f}" for i in range(buckets)]
    return dict(zip(labels, counts))


def benchmark_model(model_name: str, frames: List[np.ndarray], confidence_threshold: float,
                     device: str, warmup_frames: int) -> dict:
    cfg = Config(model=model_name, confidence_threshold=confidence_threshold, device=device)
    cfg.validate_paths()
    detector = create_detector(cfg)

    warmup_count = min(warmup_frames, len(frames))
    for frame in frames[:warmup_count]:
        detector.detect(frame)

    latencies = []
    confidences = []
    detection_counts = []
    for frame in frames:
        start = time.perf_counter()
        detections = detector.detect(frame)
        latencies.append(time.perf_counter() - start)
        detection_counts.append(len(detections))
        confidences.extend(d.confidence for d in detections)

    mean_latency = statistics.mean(latencies) if latencies else 0.0
    sorted_latencies = sorted(latencies)
    p95_latency = sorted_latencies[int(0.95 * len(sorted_latencies)) - 1] if sorted_latencies else 0.0

    # MobileNet-SSD always runs on OpenCV DNN's default (CPU) backend in this
    # project -- Phase 1/2 never wired up a CUDA backend for it -- so
    # reporting the --device flag for it would misrepresent what actually ran.
    device_used = "cpu" if model_name == "mobilenet" else device

    return {
        "model": model_name,
        "device": device_used,
        "frames": len(frames),
        "warmup_frames": warmup_count,
        "resolution": f"{frames[0].shape[1]}x{frames[0].shape[0]}" if frames else "n/a",
        "mean_latency_ms": mean_latency * 1000,
        "median_latency_ms": statistics.median(latencies) * 1000 if latencies else 0.0,
        "p95_latency_ms": p95_latency * 1000,
        "fps": 1.0 / mean_latency if mean_latency > 0 else 0.0,
        "total_detections": sum(detection_counts),
        "mean_detections_per_frame": statistics.mean(detection_counts) if detection_counts else 0.0,
        "confidence_min": min(confidences) if confidences else None,
        "confidence_max": max(confidences) if confidences else None,
        "confidence_mean": statistics.mean(confidences) if confidences else None,
        "confidence_histogram": confidence_histogram(confidences, low=confidence_threshold),
    }


def print_results(results: List[dict]) -> None:
    print("\n=== Benchmark results (speed only -- see evaluate.py for accuracy) ===")
    header = f"{'Model':<12}{'Device':<8}{'Res':<12}{'Frames':<8}{'MeanLat(ms)':<14}{'FPS':<10}{'Detections'}"
    print(header)
    for r in results:
        print(
            f"{r['model']:<12}{r['device']:<8}{r['resolution']:<12}{r['frames']:<8}"
            f"{r['mean_latency_ms']:<14.2f}{r['fps']:<10.2f}{r['total_detections']}"
        )

    for r in results:
        print(f"\n--- {r['model']} ---")
        print(f"  Warm-up frames (excluded from timing): {r['warmup_frames']}")
        print(f"  Latency: mean={r['mean_latency_ms']:.2f}ms median={r['median_latency_ms']:.2f}ms p95={r['p95_latency_ms']:.2f}ms")
        print(f"  Mean detections/frame: {r['mean_detections_per_frame']:.2f}")
        if r["confidence_mean"] is not None:
            print(f"  Confidence: min={r['confidence_min']:.3f} max={r['confidence_max']:.3f} mean={r['confidence_mean']:.3f}")
            print(f"  Confidence histogram: {r['confidence_histogram']}")
        else:
            print("  Confidence: n/a (no detections)")


def main(argv=None) -> List[dict]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="data/eval/images",
                         help="Directory of images, a video file, or a webcam index (default: data/eval/images)")
    parser.add_argument("--models", nargs="+", default=["mobilenet", "yolov8"], choices=["mobilenet", "yolov8"])
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=21)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None)
    args = parser.parse_args(argv)

    resize_to = tuple(args.resize) if args.resize else None
    frames = load_frames(args.source, args.max_frames, resize_to)
    if not frames:
        raise RuntimeError(f"No frames loaded from {args.source!r}")
    logger.info(f"Loaded {len(frames)} frames from {args.source} (same frames used for every model below)")

    results = []
    for model_name in args.models:
        logger.info(f"Benchmarking {model_name}...")
        results.append(benchmark_model(model_name, frames, args.confidence, args.device, args.warmup_frames))

    print_results(results)
    return results


if __name__ == "__main__":
    main()
