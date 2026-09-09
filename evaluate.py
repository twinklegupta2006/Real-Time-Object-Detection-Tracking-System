"""Accuracy evaluation: compares detector backends on the same small, real,
ground-truth-labeled image set (data/eval/), at the same confidence
threshold, through the common detector interface (src/detectors).

This measures *accuracy*, not speed -- see benchmark.py / README for FPS and
latency, which are a separate question from correctness.

Usage:
    python evaluate.py                              # both models, defaults
    python evaluate.py --models yolov8               # one model only
    python evaluate.py --confidence 0.6
"""
import argparse
import logging

from src.config import Config
from src.detectors import create_detector
from src.evaluation.class_names import normalize_class_name
from src.evaluation.dataset import load_eval_dataset
from src.evaluation.metrics import ScoredBox, evaluate_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_model(model_name: str, dataset_dir: str, confidence_threshold: float, device: str) -> dict:
    cfg = Config(model=model_name, confidence_threshold=confidence_threshold, device=device)
    cfg.validate_paths()
    detector = create_detector(cfg)

    dataset = load_eval_dataset(dataset_dir)

    per_image_predictions = []
    per_image_ground_truths = []
    for item in dataset:
        detections = detector.detect(item.frame)
        preds = [
            ScoredBox(
                x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                class_name=normalize_class_name(d.class_name),
                confidence=d.confidence,
            )
            for d in detections
        ]
        per_image_predictions.append(preds)
        per_image_ground_truths.append(item.ground_truth)

    return evaluate_dataset(per_image_predictions, per_image_ground_truths)


def print_results(results: dict) -> None:
    print(f"\n=== Evaluation results (IoU@0.5) ===")
    header = f"{'Model':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}{'mAP@50':<12}{'MeanIoU':<12}{'TP/FP/FN'}"
    print(header)
    for model_name, r in results.items():
        print(
            f"{model_name:<12}{r['precision']:<12.3f}{r['recall']:<12.3f}{r['f1']:<12.3f}"
            f"{r['map50']:<12.3f}{r['mean_iou']:<12.3f}{r['tp']}/{r['fp']}/{r['fn']}"
        )

    for model_name, r in results.items():
        print(f"\n--- {model_name}: per-class AP@50 ---")
        for class_name, ap in sorted(r["per_class_ap"].items()):
            print(f"  {class_name:<16}{'n/a (no GT in eval set)' if ap is None else f'{ap:.3f}'}")


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default="data/eval")
    parser.add_argument("--models", nargs="+", default=["mobilenet", "yolov8"], choices=["mobilenet", "yolov8"])
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    logger.info(f"Evaluating on dataset: {args.dataset_dir} (confidence >= {args.confidence})")
    results = {}
    for model_name in args.models:
        logger.info(f"Running {model_name}...")
        results[model_name] = run_model(model_name, args.dataset_dir, args.confidence, args.device)

    print_results(results)
    return results


if __name__ == "__main__":
    main()
