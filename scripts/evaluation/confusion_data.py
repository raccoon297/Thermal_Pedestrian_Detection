"""Build and cache detection confusion-matrix counts from trained models."""

import csv
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


EXPERIMENTS = ("baseline", "clahe", "bilateral_clahe")
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def _cache_is_current(cache_path: Path, weights_path: Path) -> bool:
    if not cache_path.is_file() or cache_path.stat().st_mtime < weights_path.stat().st_mtime:
        return False
    try:
        with cache_path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        return len(rows) == 4 and sum(int(row["count"]) for row in rows) > 0
    except (KeyError, TypeError, ValueError):
        return False


def _write_cache(path: Path, matrix: Any, class_name: str) -> None:
    labels = (class_name, "background")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("predicted", "true", "count", "confidence_threshold", "iou_threshold"),
        )
        writer.writeheader()
        for predicted_index, predicted_name in enumerate(labels):
            for true_index, true_name in enumerate(labels):
                writer.writerow({
                    "predicted": predicted_name,
                    "true": true_name,
                    "count": int(matrix[predicted_index, true_index]),
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "iou_threshold": IOU_THRESHOLD,
                })


def ensure_confusion_data(project_root: Path) -> list[Path]:
    """Validate only models whose cached confusion counts are missing or stale."""
    pending: list[tuple[str, Path, Path, Path]] = []
    cache_paths: list[Path] = []
    for experiment in EXPERIMENTS:
        result_dir = project_root / "results" / experiment
        weights_path = result_dir / "best.pt"
        dataset_yaml = project_root / "configs" / "datasets" / f"{experiment}.yaml"
        cache_path = result_dir / "confusion_matrix.csv"
        if not weights_path.is_file():
            raise FileNotFoundError(f"Missing trained weights: {weights_path}")
        if not dataset_yaml.is_file():
            raise FileNotFoundError(f"Missing dataset configuration: {dataset_yaml}")
        cache_paths.append(cache_path)
        if not _cache_is_current(cache_path, weights_path):
            pending.append((experiment, weights_path, dataset_yaml, cache_path))

    if not pending:
        return cache_paths

    config_dir = Path(tempfile.gettempdir()) / "ThermalSight_Ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("YOLO_VERBOSE", "false")
    from ultralytics import YOLO

    run_dir = project_root / ".evaluation_temp"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        for experiment, weights_path, dataset_yaml, cache_path in pending:
            print(f"[VALIDATE] {experiment} confusion matrix")
            model = YOLO(str(weights_path))
            metrics = model.val(
                data=str(dataset_yaml),
                imgsz=640,
                batch=16,
                conf=CONFIDENCE_THRESHOLD,
                device=0,
                workers=0,
                plots=True,
                save=False,
                verbose=False,
                project=run_dir,
                name=experiment,
                exist_ok=True,
            )
            matrix = metrics.confusion_matrix.matrix
            if matrix.shape != (2, 2):
                raise ValueError(
                    f"Expected a one-class 2x2 confusion matrix for {experiment}, got {matrix.shape}"
                )
            class_name = str(metrics.names[0])
            _write_cache(cache_path, matrix, class_name)
            print(f"[CACHED] {cache_path}")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    return cache_paths
