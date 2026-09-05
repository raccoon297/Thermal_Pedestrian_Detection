"""Fair, repeatable YOLO26n training for ThermalSight experiments."""

import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ULTRALYTICS_CONFIG_DIR = Path(tempfile.gettempdir()) / "ThermalSight_Ultralytics"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
os.environ.setdefault("YOLO_VERBOSE", "false")

from ultralytics import YOLO, __version__ as ULTRALYTICS_VERSION  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.build import build_dataloader, build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402

EXPERIMENTS = ("baseline", "clahe", "bilateral_clahe")
EXPECTED_COUNTS = {"train": 2110, "val": 112}
TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"
DATASET_CONFIG_DIR = PROJECT_ROOT / "configs" / "datasets"
RESULTS_ROOT = PROJECT_ROOT / "results"
TRAIN_ARGUMENT_KEYS = (
    "pretrained", "epochs", "imgsz", "batch", "seed", "workers", "device",
    "optimizer", "patience", "deterministic", "plots",
)
REQUIRED_CONFIG_KEYS = {"model", "augmentation", *TRAIN_ARGUMENT_KEYS}


def _load_training_config() -> dict[str, Any]:
    config = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != REQUIRED_CONFIG_KEYS:
        raise ValueError(
            f"training.yaml keys must be exactly {sorted(REQUIRED_CONFIG_KEYS)}"
        )
    if config["model"] != "yolo26n.pt":
        raise ValueError("The experiment model is fixed to yolo26n.pt")
    if config["augmentation"] != "default":
        raise ValueError("Only Ultralytics default augmentation is allowed")
    return config


def _validate_dataset(experiment: str, dataset_yaml: Path) -> dict[str, Any]:
    model_input_root = (
        PROJECT_ROOT / "data" / "processed" / "thermal_night"
        / "model_input" / experiment
    )
    missing_directories = [
        model_input_root / "images" / split
        for split in ("train", "val")
        if not (model_input_root / "images" / split).is_dir()
    ]
    if missing_directories:
        raise FileNotFoundError(
            "Model-input dataset is missing. Run build_processed.py first. "
            f"Missing: {missing_directories[0]}"
        )
    raw = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if raw.get("names") != {0: "person"}:
        raise ValueError(f"Dataset must contain only class 0=person: {dataset_yaml}")
    data = check_det_dataset(str(dataset_yaml), autodownload=False)
    counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    for split in ("train", "val"):
        image_dir = Path(data[split])
        label_dir = image_dir.parents[1] / "labels" / split
        counts[split] = len(list(image_dir.glob("*.png")))
        label_counts[split] = len(list(label_dir.glob("*.txt")))
        if counts[split] != EXPECTED_COUNTS[split] or label_counts[split] != EXPECTED_COUNTS[split]:
            raise RuntimeError(
                f"{experiment}/{split} count mismatch: images={counts[split]}, labels={label_counts[split]}"
            )
    sample_path = next(Path(data["train"]).glob("*.png"))
    gray = cv2.imread(str(sample_path), cv2.IMREAD_UNCHANGED)
    color = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
    is_single_channel = gray is not None and (
        gray.ndim == 2 or (gray.ndim == 3 and gray.shape[2] == 1)
    )
    if not is_single_channel or gray.dtype.name != "uint8":
        raise RuntimeError(f"Invalid grayscale source: {sample_path}")
    if color is None or color.shape[2] != 3:
        raise RuntimeError(f"OpenCV did not expand grayscale input to 3 channels: {sample_path}")
    return {"data": data, "images": counts, "labels": label_counts, "sample": str(sample_path)}


def _validate_ultralytics_batch(data: dict[str, Any], config: dict[str, Any]) -> tuple[int, ...]:
    cfg = get_cfg(overrides={
        "imgsz": config["imgsz"], "rect": False, "cache": False, "single_cls": False,
        "classes": None, "fraction": 1.0, "task": "detect", "mode": "train",
    })
    dataset = build_yolo_dataset(
        cfg, data["train"], batch=2, data=data, mode="train", rect=False, stride=32
    )
    loader = build_dataloader(dataset, batch=2, workers=0, shuffle=False, device="cpu")
    batch = next(iter(loader))
    shape = tuple(batch["img"].shape)
    if len(shape) != 4 or shape[1] != 3:
        raise RuntimeError(f"Ultralytics batch is not NCHW 3-channel: {shape}")
    return shape


def _completed_epochs(results_csv: Path) -> int:
    if not results_csv.exists():
        return 0
    with results_csv.open(encoding="utf-8-sig", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def _metric_value(metrics: Any, name: str) -> float:
    value = getattr(metrics.box, name, None)
    if value is None:
        raise RuntimeError(f"Ultralytics validation metric is unavailable: box.{name}")
    return float(value)


def _print_epoch_progress(trainer: Any) -> None:
    """Print one compact progress line after every ten completed epochs."""
    completed = trainer.epoch + 1
    total = trainer.epochs
    if completed % 10 == 0 or completed == total:
        print(f"{completed}/{total}", flush=True)


def is_training_complete(experiment: str) -> bool:
    """Return whether the essential artifacts for an experiment already exist."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}")
    output_dir = RESULTS_ROOT / experiment
    required_artifacts = (
        output_dir / "results.csv",
        output_dir / "args.yaml",
        output_dir / "training_summary.json",
        output_dir / "best.pt",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required_artifacts)


def run_training(experiment: str) -> dict[str, Any]:
    """Train one experiment from the same pretrained YOLO26n checkpoint."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}")
    config = _load_training_config()
    dataset_yaml = DATASET_CONFIG_DIR / f"{experiment}.yaml"
    dataset_info = _validate_dataset(experiment, dataset_yaml)
    batch_shape = _validate_ultralytics_batch(dataset_info["data"], config)

    if not torch.cuda.is_available() and str(config["device"]) != "cpu":
        raise RuntimeError("training.yaml requests CUDA but torch.cuda.is_available() is false")
    device_name = torch.cuda.get_device_name(int(config["device"])) if torch.cuda.is_available() else "CPU"
    output_dir = RESULTS_ROOT / experiment
    best_copy = output_dir / "best.pt"

    train_args = {key: config[key] for key in TRAIN_ARGUMENT_KEYS}
    print(f"Ultralytics batch shape: {batch_shape}")
    print(f"Training {experiment} independently from {config['model']} on {device_name}")

    with tempfile.TemporaryDirectory(prefix=f"thermalsight_{experiment}_") as temporary:
        training_dir = Path(temporary) / experiment
        train_args.update({
            "data": str(dataset_yaml),
            "project": temporary,
            "name": experiment,
            "exist_ok": True,
            "val": True,
            "save": True,
        })
        model = YOLO(config["model"])
        model.add_callback("on_train_epoch_end", _print_epoch_progress)
        metrics = model.train(**train_args)
        best_source = training_dir / "weights" / "best.pt"
        if not best_source.exists():
            raise FileNotFoundError(f"Training did not create best.pt: {best_source}")

        output_dir.mkdir(parents=True, exist_ok=True)
        for file_name in ("results.csv", "args.yaml"):
            source = training_dir / file_name
            if not source.exists():
                raise FileNotFoundError(f"Training output not found: {source}")
            shutil.copy2(source, output_dir / file_name)
        shutil.copy2(best_source, best_copy)

    summary = {
        "experiment": experiment,
        "model": config["model"],
        "epochs_configured": config["epochs"],
        "epochs_completed": _completed_epochs(output_dir / "results.csv"),
        "imgsz": config["imgsz"],
        "batch": config["batch"],
        "seed": config["seed"],
        "device": config["device"],
        "optimizer": config["optimizer"],
        "device_name": device_name,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "ultralytics": ULTRALYTICS_VERSION,
        "dataset": {
            "yaml": str(dataset_yaml),
            "train_images": dataset_info["images"]["train"],
            "val_images": dataset_info["images"]["val"],
            "train_labels": dataset_info["labels"]["train"],
            "val_labels": dataset_info["labels"]["val"],
            "grayscale_source": True,
            "ultralytics_batch_shape": list(batch_shape),
        },
        "metrics": {
            "precision": _metric_value(metrics, "mp"),
            "recall": _metric_value(metrics, "mr"),
            "map50": _metric_value(metrics, "map50"),
            "map50_95": _metric_value(metrics, "map"),
        },
        "precision": _metric_value(metrics, "mp"),
        "recall": _metric_value(metrics, "mr"),
        "map50": _metric_value(metrics, "map50"),
        "map50_95": _metric_value(metrics, "map"),
        "best_model_path": str(best_copy.resolve()),
        "dataset_yaml": str(dataset_yaml.resolve()),
        "best_pt": str(best_copy.resolve()),
        "training_output": str(output_dir.resolve()),
        "common_config_snapshot": config,
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Training summary: {summary_path}")
    return summary
