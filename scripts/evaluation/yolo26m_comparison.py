"""Train and compare Baseline+YOLO26m against CLAHE+YOLO26n.

This module intentionally keeps the user-facing entry point at the project root.
Running ``compare_yolo26m.py`` performs every missing step and writes the final
comparison figure as ``results/evaluation/07_yolo26n_clahe_vs_yolo26m.png``.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import matplotlib
import numpy as np
import torch
import yaml
from PIL import Image

from scripts.final_model import (
    FINAL_CONFIDENCE,
    FINAL_INFERENCE_IMAGE_SIZE,
    FINAL_PERSON_CLASS_ID,
    FINAL_TRAINING_IMAGE_SIZE,
    REFERENCE_INFERENCE_IMAGE_SIZE,
)
from scripts.preprocessing.thermal import preprocess_baseline, preprocess_clahe_optimized


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


MODEL_NAME = "yolo26m.pt"
REFERENCE_EXPERIMENT = "yolo26m_baseline"
TRAINING_IMAGE_SIZE = FINAL_TRAINING_IMAGE_SIZE
PROPOSED_IMAGE_SIZE = FINAL_INFERENCE_IMAGE_SIZE
REFERENCE_IMAGE_SIZE = REFERENCE_INFERENCE_IMAGE_SIZE
EXPECTED_COUNTS = {"train": 2110, "val": 112}
EXPECTED_THERMAL_SHAPE = (512, 640)
BENCHMARK_WARMUP = 20
BENCHMARK_REPEATS = 3
MODEL_COLORS = {"proposed": "#12B981", "reference": "#3B82F6"}


@dataclass(frozen=True)
class ComparisonTarget:
    key: str
    display_name: str
    preprocessing_name: str
    weights: Path
    dataset_yaml: Path
    preprocessor: Callable[[np.ndarray], np.ndarray]
    image_size: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_dir() -> Path:
    """Keep disposable validation inputs outside the project workspace."""
    runtime_dir = Path(tempfile.gettempdir()) / "ThermalSight_Comparison"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _configure_ultralytics() -> None:
    config_dir = _runtime_dir() / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("YOLO_VERBOSE", "false")


def _load_training_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "configs" / "training.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "model", "pretrained", "epochs", "imgsz", "batch", "seed", "workers",
        "device", "optimizer", "patience", "deterministic", "augmentation", "plots",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"Unexpected training configuration keys: {path}")
    if config["imgsz"] != TRAINING_IMAGE_SIZE:
        raise ValueError(f"Stored training protocol requires imgsz={TRAINING_IMAGE_SIZE}")
    if config["augmentation"] != "default":
        raise ValueError("Comparison requires the shared default augmentation setting")
    return config


def _dataset_counts(dataset_yaml: Path) -> dict[str, int]:
    raw = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if raw.get("names") != {0: "person"}:
        raise ValueError(f"Dataset must contain only class 0=person: {dataset_yaml}")
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        image_dir = (dataset_yaml.parent / raw[split]).resolve()
        label_dir = image_dir.parents[1] / "labels" / split
        image_count = len(list(image_dir.glob("*.png")))
        label_count = len(list(label_dir.glob("*.txt")))
        if image_count != EXPECTED_COUNTS[split] or label_count != EXPECTED_COUNTS[split]:
            raise RuntimeError(
                f"{dataset_yaml.stem}/{split} count mismatch: "
                f"images={image_count}, labels={label_count}"
            )
        counts[split] = image_count
    return counts


def _completed_epochs(results_csv: Path) -> int:
    with results_csv.open(encoding="utf-8-sig", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def _metric_value(metrics: Any, name: str) -> float:
    value = getattr(metrics.box, name, None)
    if value is None:
        raise RuntimeError(f"Ultralytics metric box.{name} is unavailable")
    return float(value)


def _print_epoch_progress(trainer: Any) -> None:
    completed = trainer.epoch + 1
    if completed == 1 or completed % 5 == 0 or completed == trainer.epochs:
        print(f"  epoch {completed}/{trainer.epochs}", flush=True)


def _reference_is_complete(project_root: Path, config: dict[str, Any]) -> bool:
    output_dir = project_root / "results" / REFERENCE_EXPERIMENT
    summary_path = output_dir / "training_summary.json"
    required = (
        output_dir / "best.pt", output_dir / "results.csv",
        output_dir / "args.yaml", summary_path,
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_fingerprints = {
        "training_config": _sha256(project_root / "configs" / "training.yaml"),
        "dataset_config": _sha256(project_root / "configs" / "datasets" / "baseline.yaml"),
    }
    return (
        summary.get("model") == MODEL_NAME
        and summary.get("epochs_completed") == config["epochs"]
        and summary.get("fingerprints") == expected_fingerprints
    )


def _ensure_base_checkpoint(project_root: Path) -> Path:
    checkpoint = project_root / MODEL_NAME
    if checkpoint.is_file():
        return checkpoint
    print(f"\n[DOWNLOAD] {MODEL_NAME}")
    from ultralytics import YOLO

    previous = Path.cwd()
    try:
        os.chdir(project_root)
        YOLO(MODEL_NAME)
    finally:
        os.chdir(previous)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Ultralytics did not create {checkpoint}")
    return checkpoint


def _train_reference(project_root: Path, config: dict[str, Any]) -> Path:
    output_dir = project_root / "results" / REFERENCE_EXPERIMENT
    if _reference_is_complete(project_root, config):
        print(f"\n[SKIP] Reusing completed {REFERENCE_EXPERIMENT} training artifacts")
        return output_dir / "best.pt"

    dataset_yaml = project_root / "configs" / "datasets" / "baseline.yaml"
    counts = _dataset_counts(dataset_yaml)
    base_checkpoint = _ensure_base_checkpoint(project_root)
    from ultralytics import YOLO, __version__ as ultralytics_version

    train_keys = (
        "pretrained", "epochs", "imgsz", "batch", "seed", "workers", "device",
        "optimizer", "patience", "deterministic", "plots",
    )
    train_args = {key: config[key] for key in train_keys}
    print(
        f"\n[TRAIN] Baseline normalization + YOLO26m\n"
        f"  train={counts['train']:,}, val={counts['val']:,}, "
        f"epochs={config['epochs']}, batch={config['batch']}, imgsz={config['imgsz']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(base_checkpoint))
    model.add_callback("on_train_epoch_end", _print_epoch_progress)
    metrics = model.train(
        **train_args,
        data=str(dataset_yaml),
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        val=True,
        save=True,
    )
    best_source = output_dir / "weights" / "best.pt"
    for source in (best_source, output_dir / "results.csv", output_dir / "args.yaml"):
        if not source.is_file():
            raise FileNotFoundError(f"Training artifact not found: {source}")
    shutil.copy2(best_source, output_dir / "best.pt")

    summary = {
        "experiment": REFERENCE_EXPERIMENT,
        "model": MODEL_NAME,
        "preprocessing": "baseline",
        "epochs_configured": config["epochs"],
        "epochs_completed": _completed_epochs(output_dir / "results.csv"),
        "dataset": {"train_images": counts["train"], "val_images": counts["val"]},
        "metrics": {
            "precision": _metric_value(metrics, "mp"),
            "recall": _metric_value(metrics, "mr"),
            "map50": _metric_value(metrics, "map50"),
            "map50_95": _metric_value(metrics, "map"),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics_version,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "common_config_snapshot": config,
        "fingerprints": {
            "training_config": _sha256(project_root / "configs" / "training.yaml"),
            "dataset_config": _sha256(dataset_yaml),
        },
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] YOLO26m best weight: {output_dir / 'best.pt'}")
    return output_dir / "best.pt"


def _comparison_targets(project_root: Path, reference_weights: Path) -> tuple[ComparisonTarget, ...]:
    return (
        ComparisonTarget(
            key="proposed",
            display_name="Optimized CLAHE + YOLO26n (960)",
            preprocessing_name="optimized_clahe",
            weights=project_root / "results" / "clahe" / "best.pt",
            dataset_yaml=project_root / "configs" / "datasets" / "clahe.yaml",
            preprocessor=preprocess_clahe_optimized,
            image_size=PROPOSED_IMAGE_SIZE,
        ),
        ComparisonTarget(
            key="reference",
            display_name="Baseline + YOLO26m (640)",
            preprocessing_name="baseline",
            weights=reference_weights,
            dataset_yaml=project_root / "configs" / "datasets" / "baseline.yaml",
            preprocessor=preprocess_baseline,
            image_size=REFERENCE_IMAGE_SIZE,
        ),
    )


def _runtime_validation_yaml(
    project_root: Path, target: ComparisonTarget, temporary_root: Path
) -> Path:
    """Build every validation input in the disposable system-temp workspace."""
    dataset_root = temporary_root / f"{target.key}_{target.preprocessing_name}_dataset"
    image_dir = dataset_root / "images" / "val"
    label_dir = dataset_root / "labels" / "val"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    source_dir = (
        project_root / "data" / "processed" / "thermal_night" / "source_tiff" / "val"
    )
    source_labels = (
        project_root / "data" / "processed" / "thermal_night"
        / "model_input" / target.dataset_yaml.stem / "labels" / "val"
    )
    source_paths = sorted(source_dir.glob("*.tiff"))
    if len(source_paths) != EXPECTED_COUNTS["val"]:
        raise RuntimeError(
            f"Expected {EXPECTED_COUNTS['val']} raw validation TIFFs, found {len(source_paths)}"
        )
    for source_path in source_paths:
        with Image.open(source_path) as image:
            thermal = np.asarray(image).copy()
        processed = target.preprocessor(thermal)
        output_path = image_dir / f"{source_path.stem}.png"
        if not cv2.imwrite(str(output_path), processed):
            raise RuntimeError(f"Failed to write runtime validation image: {output_path}")
        source_label = source_labels / f"{source_path.stem}.txt"
        if not source_label.is_file():
            raise FileNotFoundError(f"Validation label is missing: {source_label}")
        shutil.copy2(source_label, label_dir / source_label.name)

    runtime_yaml = dataset_root / "dataset.yaml"
    runtime_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root.resolve()),
                "train": "images/val",
                "val": "images/val",
                "names": {0: "person"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return runtime_yaml


def _validate_target(
    project_root: Path, target: ComparisonTarget, temporary_root: Path
) -> dict[str, float]:
    from ultralytics import YOLO

    print(f"\n[VALIDATE] {target.display_name}")
    (temporary_root / f"validation_{target.key}").mkdir(parents=True, exist_ok=True)
    validation_yaml = _runtime_validation_yaml(project_root, target, temporary_root)
    model = YOLO(str(target.weights))
    metrics = model.val(
        data=str(validation_yaml),
        imgsz=target.image_size,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=0,
        plots=False,
        save=False,
        verbose=False,
        project=temporary_root,
        name=f"validation_{target.key}",
        exist_ok=True,
    )
    values = {
        "precision": _metric_value(metrics, "mp"),
        "recall": _metric_value(metrics, "mr"),
        "map50": _metric_value(metrics, "map50"),
        "map50_95": _metric_value(metrics, "map"),
    }
    print(
        "  " + ", ".join(f"{name}={value:.4f}" for name, value in values.items())
    )
    del model, metrics
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return values


def _load_validation_thermal(project_root: Path) -> list[np.ndarray]:
    paths = sorted(
        (project_root / "data" / "processed" / "thermal_night" / "source_tiff" / "val").glob("*.tiff")
    )
    if len(paths) != EXPECTED_COUNTS["val"]:
        raise RuntimeError(f"Expected 112 validation TIFFs, found {len(paths)}")
    images: list[np.ndarray] = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image).copy()
        if array.dtype != np.uint16 or array.shape != EXPECTED_THERMAL_SHAPE:
            raise ValueError(f"Invalid thermal TIFF: {path}")
        images.append(array)
    return images


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _benchmark_target(target: ComparisonTarget, thermal_images: list[np.ndarray]) -> dict[str, float | int]:
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n[BENCHMARK] {target.display_name}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = YOLO(str(target.weights))
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    gflops = float(get_flops(model.model, imgsz=target.image_size))

    warm = cv2.cvtColor(target.preprocessor(thermal_images[0]), cv2.COLOR_GRAY2BGR)
    for _ in range(BENCHMARK_WARMUP):
        model.predict(
            source=warm, imgsz=target.image_size, conf=FINAL_CONFIDENCE,
            classes=[FINAL_PERSON_CLASS_ID],
            device=device, verbose=False, save=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    end_to_end_ms: list[float] = []
    for repeat in range(BENCHMARK_REPEATS):
        for thermal in thermal_images:
            start = time.perf_counter()
            processed = target.preprocessor(thermal)
            preprocessing_done = time.perf_counter()
            model_input = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_start = time.perf_counter()
            model.predict(
                source=model_input, imgsz=target.image_size, conf=FINAL_CONFIDENCE,
                classes=[FINAL_PERSON_CLASS_ID],
                device=device, verbose=False, save=False,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            preprocess_ms.append((preprocessing_done - start) * 1000.0)
            inference_ms.append((end - inference_start) * 1000.0)
            end_to_end_ms.append((end - start) * 1000.0)
        print(f"  repeat {repeat + 1}/{BENCHMARK_REPEATS}", flush=True)

    peak_allocated = (
        torch.cuda.max_memory_allocated() / (1024.0 ** 2) if torch.cuda.is_available() else 0.0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved() / (1024.0 ** 2) if torch.cuda.is_available() else 0.0
    )
    result: dict[str, float | int] = {
        "samples": len(end_to_end_ms),
        "warmup": BENCHMARK_WARMUP,
        "inference_imgsz": target.image_size,
        "parameters": int(parameters),
        "gflops": gflops,
        "checkpoint_mb": target.weights.stat().st_size / (1024.0 ** 2),
        "preprocess_mean_ms": statistics.fmean(preprocess_ms),
        "preprocess_p95_ms": _percentile(preprocess_ms, 95),
        "inference_mean_ms": statistics.fmean(inference_ms),
        "inference_p95_ms": _percentile(inference_ms, 95),
        "end_to_end_mean_ms": statistics.fmean(end_to_end_ms),
        "end_to_end_median_ms": statistics.median(end_to_end_ms),
        "end_to_end_p95_ms": _percentile(end_to_end_ms, 95),
        "fps": 1000.0 / statistics.fmean(end_to_end_ms),
        "peak_vram_allocated_mb": peak_allocated,
        "peak_vram_reserved_mb": peak_reserved,
    }
    print(
        f"  params={parameters / 1e6:.2f}M, GFLOPs={gflops:.2f}, "
        f"E2E={result['end_to_end_mean_ms']:.2f}ms, "
        f"FPS={result['fps']:.1f}, VRAM={peak_allocated:.1f}MB"
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _reduction(smaller: float, larger: float) -> float:
    return (1.0 - smaller / larger) * 100.0


def _write_csv(path: Path, results: dict[str, dict[str, Any]]) -> None:
    fields = [
        "key", "display_name", "preprocessing", "inference_imgsz", "precision", "recall", "map50", "map50_95",
        "parameters", "gflops", "checkpoint_mb", "preprocess_mean_ms", "inference_mean_ms",
        "end_to_end_mean_ms", "end_to_end_p95_ms", "fps", "peak_vram_allocated_mb",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for key, result in results.items():
            row = {field: result.get(field, "") for field in fields}
            row["key"] = key
            writer.writerow(row)


def _create_figure(path: Path, results: dict[str, dict[str, Any]], device_name: str) -> None:
    proposed = results["proposed"]
    reference = results["reference"]
    labels = [proposed["display_name"], reference["display_name"]]
    colors = [MODEL_COLORS["proposed"], MODEL_COLORS["reference"]]

    fig = plt.figure(figsize=(16, 9), facecolor="#F7F9FC")
    grid = fig.add_gridspec(2, 3, left=0.06, right=0.97, top=0.84, bottom=0.09, hspace=0.38, wspace=0.30)
    fig.suptitle(
        "YOLO26n and YOLO26m System Comparison",
        fontsize=24, fontweight="bold", color="#172033", y=0.965,
    )
    fig.text(
        0.5, 0.91,
        f"Optimized CLAHE + YOLO26n (960) vs Baseline + YOLO26m (640)  |  "
        f"FLIR Night Validation (112 images)  |  {device_name}",
        ha="center", fontsize=11, color="#5D6B82",
    )

    axis = fig.add_subplot(grid[0, :2])
    metric_keys = ("precision", "recall", "map50", "map50_95")
    metric_names = ("Precision", "Recall", "mAP50", "mAP50-95")
    x = np.arange(len(metric_keys))
    width = 0.34
    for index, result in enumerate((proposed, reference)):
        values = [result[key] for key in metric_keys]
        bars = axis.bar(x + (index - 0.5) * width, values, width, color=colors[index], label=labels[index])
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
    axis.set_xticks(x, metric_names)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Detection metric")
    axis.set_title("Detection Performance", loc="left", fontsize=15, fontweight="bold")
    axis.legend(frameon=False, ncols=2, loc="lower center")
    axis.grid(axis="y", alpha=0.18)

    axis = fig.add_subplot(grid[0, 2])
    efficiency_keys = ("parameters", "gflops", "checkpoint_mb", "peak_vram_allocated_mb", "end_to_end_mean_ms")
    efficiency_names = ("Params", "GFLOPs", "Weight", "VRAM", "E2E")
    ratios = [proposed[key] / reference[key] * 100.0 for key in efficiency_keys]
    bars = axis.barh(efficiency_names, ratios, color=MODEL_COLORS["proposed"])
    axis.axvline(100, color=MODEL_COLORS["reference"], linestyle="--", linewidth=1.5, label="YOLO26m = 100%")
    for bar, value in zip(bars, ratios):
        if value >= 82:
            axis.text(
                value - 2.0, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%", ha="right", va="center",
                fontsize=9, fontweight="bold", color="white",
            )
        else:
            axis.text(
                value + 1.6, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%", ha="left", va="center",
                fontsize=9, color="#172033",
            )
    axis.set_xlim(0, 116)
    axis.set_xlabel("Relative resource use")
    axis.set_title("Resource Use Relative to YOLO26m", loc="left", fontsize=15, fontweight="bold")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="x", alpha=0.18)

    axis = fig.add_subplot(grid[1, 0])
    preprocess = [proposed["preprocess_mean_ms"], reference["preprocess_mean_ms"]]
    inference = [proposed["inference_mean_ms"], reference["inference_mean_ms"]]
    positions = np.arange(2)
    axis.bar(positions, preprocess, color="#F59E0B", label="Thermal preprocessing")
    axis.bar(positions, inference, bottom=preprocess, color="#64748B", label="Model + postprocess")
    axis.set_xticks(positions, ("YOLO26n\n(960)", "YOLO26m\n(640)"))
    maximum_latency = max(p + i for p, i in zip(preprocess, inference))
    axis.set_ylim(0, maximum_latency * 1.24)
    axis.set_ylabel("Mean latency (ms)")
    axis.set_title("Batch-1 Latency Breakdown", loc="left", fontsize=15, fontweight="bold")
    axis.legend(
        frameon=False, fontsize=8, ncols=2, loc="upper center",
        bbox_to_anchor=(0.5, 0.995), borderaxespad=0.0,
        handlelength=1.8, columnspacing=1.2,
    )
    axis.grid(axis="y", alpha=0.18)

    axis = fig.add_subplot(grid[1, 1])
    resource_names = ("Parameters (M)", "GFLOPs", "Checkpoint (MB)", "Peak VRAM (MB)")
    resource_keys = ("parameters", "gflops", "checkpoint_mb", "peak_vram_allocated_mb")
    rows = []
    for name, key in zip(resource_names, resource_keys):
        divisor = 1e6 if key == "parameters" else 1.0
        rows.append([
            name,
            f"{proposed[key] / divisor:.2f}",
            f"{reference[key] / divisor:.2f}",
            f"-{_reduction(proposed[key], reference[key]):.1f}%",
        ])
    axis.axis("off")
    axis.set_title("Compute and Memory", loc="left", fontsize=15, fontweight="bold", pad=8)
    table = axis.table(
        cellText=rows, colLabels=("Metric", "YOLO26n\n(960)", "YOLO26m\n(640)", "Reduction"),
        cellLoc="center", colLoc="center", colWidths=(0.34, 0.23, 0.25, 0.20),
        bbox=(0, 0.04, 1, 0.82),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#DCE3ED")
        if row == 0:
            cell.set_facecolor("#E8EEF7")
            cell.set_text_props(weight="bold", color="#172033")
        elif column == 3:
            cell.set_text_props(weight="bold", color=MODEL_COLORS["proposed"])

    axis = fig.add_subplot(grid[1, 2])
    axis.axis("off")
    recall_delta = (proposed["recall"] - reference["recall"]) * 100.0
    map_delta = (proposed["map50_95"] - reference["map50_95"]) * 100.0
    conclusion = (
        f"Recall change\n{recall_delta:+.2f}%p\n\n"
        f"mAP50-95 change\n{map_delta:+.2f}%p\n\n"
        f"End-to-end latency reduction\n"
        f"{_reduction(proposed['end_to_end_mean_ms'], reference['end_to_end_mean_ms']):.1f}%\n\n"
        f"Throughput\n{proposed['fps']:.1f} vs {reference['fps']:.1f} FPS"
    )
    axis.text(
        0.5, 0.43, conclusion, ha="center", va="center", fontsize=12.5, linespacing=1.32,
        color="#172033", bbox={"boxstyle": "round,pad=0.82", "facecolor": "white", "edgecolor": "#DCE3ED"},
    )
    axis.set_title(
        "Performance Difference", loc="left", fontsize=15, fontweight="bold", pad=14
    )

    fig.text(
        0.5, 0.025,
        f"Latency: {BENCHMARK_WARMUP} warm-up runs, {EXPECTED_COUNTS['val']} images × {BENCHMARK_REPEATS} repeats, "
        "batch=1, CUDA synchronized, disk I/O excluded. Peak VRAM is allocated CUDA memory.",
        ha="center", fontsize=9, color="#6B7280",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_comparison(project_root: Path) -> dict[str, dict[str, Any]]:
    """Run the missing training, validation, benchmark, and report generation."""
    project_root = project_root.resolve()
    _configure_ultralytics()
    if not torch.cuda.is_available():
        raise RuntimeError("This comparison is configured for CUDA, but CUDA is unavailable")
    config = _load_training_config(project_root)
    reference_weights = _train_reference(project_root, config)
    targets = _comparison_targets(project_root, reference_weights)
    for target in targets:
        if not target.weights.is_file():
            raise FileNotFoundError(f"Model weight not found: {target.weights}")
        _dataset_counts(target.dataset_yaml)

    validation_root = _runtime_dir() / "validation"
    validation_root.mkdir(parents=True, exist_ok=True)
    metrics = {
        target.key: _validate_target(project_root, target, validation_root)
        for target in targets
    }
    thermal_images = _load_validation_thermal(project_root)
    benchmarks = {
        target.key: _benchmark_target(target, thermal_images) for target in targets
    }
    results: dict[str, dict[str, Any]] = {}
    for target in targets:
        results[target.key] = {
            "display_name": target.display_name,
            "preprocessing": target.preprocessing_name,
            "weights": str(target.weights),
            **metrics[target.key],
            **benchmarks[target.key],
        }

    output_dir = project_root / "results" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "comparison": results,
        "protocol": {
            "training_image_size": TRAINING_IMAGE_SIZE,
            "inference_image_sizes": {
                target.key: target.image_size for target in targets
            },
            "comparison_scope": "deployment-system operating points; model-only comparison is not claimed",
            "proposed_preprocessing_implementation": (
                "uint16 histogram percentile normalization + cached OpenCV CLAHE"
            ),
            "validation_source": "all targets built from raw uint16 TIFF at evaluation time",
            "ablation_dataset_preserved": True,
            "batch": 1,
            "validation_images": EXPECTED_COUNTS["val"],
            "warmup": BENCHMARK_WARMUP,
            "repeats": BENCHMARK_REPEATS,
            "disk_io_included": False,
            "cuda_synchronized": True,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
    }
    json_path = output_dir / "07_yolo26n_clahe_vs_yolo26m.json"
    csv_path = output_dir / "07_yolo26n_clahe_vs_yolo26m.csv"
    figure_path = output_dir / "07_yolo26n_clahe_vs_yolo26m.png"
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, results)
    _create_figure(figure_path, results, torch.cuda.get_device_name(0))
    print(
        f"\n[COMPLETE]\n  Figure: {figure_path}\n  Metrics: {json_path}\n  Table: {csv_path}"
    )
    return results
