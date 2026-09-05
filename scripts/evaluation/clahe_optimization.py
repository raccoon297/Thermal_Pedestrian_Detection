"""Compare CLAHE + YOLO26n (640) with optimized CLAHE + YOLO26n (960)."""

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

from scripts.evaluation.yolo26m_comparison import (
    ComparisonTarget,
    _benchmark_target,
    _configure_ultralytics,
    _dataset_counts,
    _load_validation_thermal,
    _runtime_dir,
    _runtime_validation_yaml,
    _validate_target,
)
from scripts.evaluation.confusion_data import CONFIDENCE_THRESHOLD, IOU_THRESHOLD
from scripts.final_model import FINAL_INFERENCE_IMAGE_SIZE, FINAL_TRAINING_IMAGE_SIZE
from scripts.preprocessing.thermal import preprocess_clahe, preprocess_clahe_optimized


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


ORIGINAL_COLOR = "#64748B"
PROPOSED_COLOR = "#12B981"


def _targets(project_root: Path) -> tuple[ComparisonTarget, ComparisonTarget]:
    weights = project_root / "results" / "clahe" / "best.pt"
    dataset_yaml = project_root / "configs" / "datasets" / "clahe.yaml"
    return (
        ComparisonTarget(
            key="original",
            display_name="CLAHE + YOLO26n (640)",
            preprocessing_name="clahe",
            weights=weights,
            dataset_yaml=dataset_yaml,
            preprocessor=preprocess_clahe,
            image_size=FINAL_TRAINING_IMAGE_SIZE,
        ),
        ComparisonTarget(
            key="proposed",
            display_name="Optimized CLAHE + YOLO26n (960)",
            preprocessing_name="optimized_clahe",
            weights=weights,
            dataset_yaml=dataset_yaml,
            preprocessor=preprocess_clahe_optimized,
            image_size=FINAL_INFERENCE_IMAGE_SIZE,
        ),
    )


def _delta_percent_points(proposed: dict[str, Any], original: dict[str, Any], key: str) -> float:
    return (proposed[key] - original[key]) * 100.0


def _change_percent(proposed: float, original: float) -> float:
    return (proposed / original - 1.0) * 100.0


def _confusion_for_target(
    project_root: Path, target: ComparisonTarget, validation_root: Path
) -> dict[str, int]:
    """Calculate a fixed-threshold one-class detection confusion matrix."""
    from ultralytics import YOLO

    validation_yaml = _runtime_validation_yaml(project_root, target, validation_root)
    model = YOLO(str(target.weights))
    metrics = model.val(
        data=str(validation_yaml),
        imgsz=target.image_size,
        batch=16,
        conf=CONFIDENCE_THRESHOLD,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=0,
        plots=True,
        save=False,
        verbose=False,
        project=validation_root,
        name=f"confusion_{target.key}",
        exist_ok=True,
    )
    matrix = np.rint(metrics.confusion_matrix.matrix).astype(np.int64)
    if matrix.shape != (2, 2):
        raise ValueError(f"Expected a one-class 2x2 confusion matrix, got {matrix.shape}")
    values = {
        "true_positive": int(matrix[0, 0]),
        "false_positive": int(matrix[0, 1]),
        "false_negative": int(matrix[1, 0]),
        "true_negative": int(matrix[1, 1]),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
    }
    del model, metrics
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return values


def _write_csv(path: Path, results: dict[str, dict[str, Any]]) -> None:
    fields = (
        "key", "display_name", "preprocessing", "inference_imgsz", "precision",
        "recall", "map50", "map50_95", "parameters", "gflops", "checkpoint_mb",
        "preprocess_mean_ms", "inference_mean_ms", "end_to_end_mean_ms",
        "end_to_end_p95_ms", "fps", "peak_vram_allocated_mb",
        "true_positive", "false_positive", "false_negative",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for key, result in results.items():
            row = {field: result.get(field, "") for field in fields}
            row["key"] = key
            row.update({
                field: result["confusion_matrix"][field]
                for field in ("true_positive", "false_positive", "false_negative")
            })
            writer.writerow(row)


def _create_figure(path: Path, results: dict[str, dict[str, Any]], device_name: str) -> None:
    original = results["original"]
    proposed = results["proposed"]

    fig = plt.figure(figsize=(16, 9), facecolor="#F7F9FC")
    grid = fig.add_gridspec(
        2, 6, left=0.055, right=0.97, top=0.83, bottom=0.11,
        height_ratios=(1.0, 1.02), hspace=0.34, wspace=0.60,
    )
    fig.suptitle(
        "CLAHE Optimization and Input Resolution Comparison",
        fontsize=24, fontweight="bold", color="#172033", y=0.965,
    )
    fig.text(
        0.5, 0.91,
        "CLAHE + YOLO26n (640)  vs  Optimized CLAHE + YOLO26n (960)"
        f"  |  Same YOLO26n checkpoint  |  FLIR Night Validation (112 images)  |  {device_name}",
        ha="center", fontsize=10.5, color="#5D6B82",
    )

    def draw_confusion(axis: Any, result: dict[str, Any], title: str, cmap: str) -> None:
        confusion = result["confusion_matrix"]
        matrix = np.array([
            [confusion["true_positive"], confusion["false_positive"]],
            [confusion["false_negative"], confusion["true_negative"]],
        ])
        axis.imshow(matrix, cmap=cmap, vmin=0, vmax=max(1, int(matrix.max())))
        annotations = (
            ("TP", confusion["true_positive"]),
            ("FP", confusion["false_positive"]),
            ("FN", confusion["false_negative"]),
            ("N/A", None),
        )
        for index, (label, value) in enumerate(annotations):
            row, column = divmod(index, 2)
            value_text = "—" if value is None else f"{value:,}"
            color = "white" if value is not None and value > matrix.max() * 0.48 else "#172033"
            axis.text(
                column, row, f"{label}\n{value_text}",
                ha="center", va="center", fontsize=15,
                fontweight="bold", color=color, linespacing=1.40,
            )
        axis.set_xticks((0, 1), ("Person", "Background"))
        axis.set_yticks((0, 1), ("Person", "Background"))
        axis.set_xlabel("Ground truth")
        axis.set_ylabel("Prediction")
        axis.set_title(title, fontsize=16, fontweight="bold", pad=12)
        for spine in axis.spines.values():
            spine.set_edgecolor("#CBD5E1")

    draw_confusion(
        fig.add_subplot(grid[0, 0:2]), original,
        "CLAHE + YOLO26n (640)", "Blues",
    )
    draw_confusion(
        fig.add_subplot(grid[0, 2:4]), proposed,
        "Optimized CLAHE + YOLO26n (960)", "Greens",
    )

    axis = fig.add_subplot(grid[1, 0:2])
    preprocess = [original["preprocess_mean_ms"], proposed["preprocess_mean_ms"]]
    inference = [original["inference_mean_ms"], proposed["inference_mean_ms"]]
    positions = np.arange(2)
    axis.bar(positions, preprocess, color="#F59E0B", label="Thermal preprocessing")
    axis.bar(
        positions, inference, bottom=preprocess,
        color="#64748B", label="Model + postprocess",
    )
    axis.set_xticks(
        positions,
        ("CLAHE + YOLO26n\n(640)", "Optimized CLAHE + YOLO26n\n(960)"),
    )
    maximum = max(p + i for p, i in zip(preprocess, inference))
    axis.set_ylim(0, maximum * 1.26)
    axis.set_ylabel("Mean latency (ms)")
    axis.set_title("Batch-1 Latency Breakdown", loc="left", fontsize=14, fontweight="bold")
    axis.legend(
        frameon=False, fontsize=7.5, ncols=2, loc="upper center",
        bbox_to_anchor=(0.5, 0.99), borderaxespad=0.0,
        handlelength=1.5, columnspacing=0.9,
    )
    axis.grid(axis="y", alpha=0.18)

    axis = fig.add_subplot(grid[1, 2:4])
    ratio_names = ("Preprocess", "E2E", "VRAM", "GFLOPs")
    ratio_keys = (
        "preprocess_mean_ms", "end_to_end_mean_ms",
        "peak_vram_allocated_mb", "gflops",
    )
    ratios = [proposed[key] / original[key] * 100.0 for key in ratio_keys]
    ratio_colors = [PROPOSED_COLOR, PROPOSED_COLOR, "#F59E0B", "#F59E0B"]
    bars = axis.barh(ratio_names, ratios, color=ratio_colors)
    axis.axvline(100, color="#3B82F6", linestyle="--", linewidth=1.5, label="640 input = 100%")
    for bar, value in zip(bars, ratios):
        axis.text(
            value - 3 if value > 55 else value + 3,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            ha="right" if value > 55 else "left", va="center",
            fontsize=8.5, fontweight="bold",
            color="white" if value > 55 else "#172033",
        )
    axis.set_xlim(0, max(ratios) * 1.12)
    axis.set_xlabel("Relative use (640 input = 100%)")
    axis.set_title("Resource Use: 960 vs 640", loc="left", fontsize=14, fontweight="bold")
    axis.legend(frameon=False, fontsize=7.5, loc="lower right")
    axis.grid(axis="x", alpha=0.18)

    rows = (
        ("Precision", f"{original['precision']:.4f}", f"{proposed['precision']:.4f}",
         f"{_delta_percent_points(proposed, original, 'precision'):+.2f}%p"),
        ("Recall", f"{original['recall']:.4f}", f"{proposed['recall']:.4f}",
         f"{_delta_percent_points(proposed, original, 'recall'):+.2f}%p"),
        ("mAP50", f"{original['map50']:.4f}", f"{proposed['map50']:.4f}",
         f"{_delta_percent_points(proposed, original, 'map50'):+.2f}%p"),
        ("mAP50-95", f"{original['map50_95']:.4f}", f"{proposed['map50_95']:.4f}",
         f"{_delta_percent_points(proposed, original, 'map50_95'):+.2f}%p"),
        ("Preprocess", f"{original['preprocess_mean_ms']:.2f} ms",
         f"{proposed['preprocess_mean_ms']:.2f} ms",
         f"{_change_percent(proposed['preprocess_mean_ms'], original['preprocess_mean_ms']):+.1f}%"),
        ("End-to-end", f"{original['end_to_end_mean_ms']:.2f} ms",
         f"{proposed['end_to_end_mean_ms']:.2f} ms",
         f"{_change_percent(proposed['end_to_end_mean_ms'], original['end_to_end_mean_ms']):+.1f}%"),
        ("GFLOPs", f"{original['gflops']:.2f}", f"{proposed['gflops']:.2f}",
         f"{_change_percent(proposed['gflops'], original['gflops']):+.1f}%"),
        ("Peak VRAM", f"{original['peak_vram_allocated_mb']:.2f} MB",
         f"{proposed['peak_vram_allocated_mb']:.2f} MB",
         f"{_change_percent(proposed['peak_vram_allocated_mb'], original['peak_vram_allocated_mb']):+.1f}%"),
    )
    axis = fig.add_subplot(grid[:, 4:6])
    axis.axis("off")
    axis.set_title("Performance Summary", loc="left", fontsize=14, fontweight="bold", pad=8)
    table = axis.table(
        cellText=rows,
        colLabels=("Metric", "CLAHE\n(640)", "Optimized CLAHE\n(960)", "Change"),
        cellLoc="center", colLoc="center",
        colWidths=(0.28, 0.25, 0.28, 0.20), bbox=(0, 0.18, 1, 0.70),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#DCE3ED")
        if row == 0:
            cell.set_facecolor("#E8EEF7")
            cell.set_text_props(weight="bold", color="#172033")
        elif row % 2 == 0:
            cell.set_facecolor("#FAFBFD")
        if column == 3 and 0 < row <= 6:
            cell.set_text_props(weight="bold", color=PROPOSED_COLOR)
        elif column == 3 and row in {7, 8}:
            cell.set_text_props(weight="bold", color="#D97706")
    axis.text(
        0.5, 0.10,
        f"FN {original['confusion_matrix']['false_negative']} → "
        f"{proposed['confusion_matrix']['false_negative']}   |   "
        f"FP {original['confusion_matrix']['false_positive']} → "
        f"{proposed['confusion_matrix']['false_positive']}",
        ha="center", va="center", fontsize=10.5,
        fontweight="bold", color="#172033",
    )

    fig.text(
        0.5, 0.018,
        f"Confusion matrices: confidence={CONFIDENCE_THRESHOLD:.2f}, IoU={IOU_THRESHOLD:.2f}.  "
        "Both settings use the same CLAHE-trained YOLO26n checkpoint. "
        "Latency: 20 warm-ups, 112 images × 3 repeats, "
        "batch=1, CUDA synchronized, disk I/O excluded.",
        ha="center", fontsize=9, color="#6B7280",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_optimization_comparison(project_root: Path) -> dict[str, dict[str, Any]]:
    """Validate and benchmark the original and final CLAHE operating points."""
    project_root = project_root.resolve()
    _configure_ultralytics()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark is configured for CUDA, but CUDA is unavailable")

    targets = _targets(project_root)
    for target in targets:
        if not target.weights.is_file():
            raise FileNotFoundError(f"CLAHE weight not found: {target.weights}")
        _dataset_counts(target.dataset_yaml)

    validation_root = _runtime_dir() / "clahe_optimization_validation"
    validation_root.mkdir(parents=True, exist_ok=True)
    metrics = {
        target.key: _validate_target(project_root, target, validation_root)
        for target in targets
    }
    print("\n[CONFUSION] Fixed confidence and IoU thresholds")
    confusions = {
        target.key: _confusion_for_target(project_root, target, validation_root)
        for target in targets
    }
    thermal_images = _load_validation_thermal(project_root)
    benchmarks = {
        target.key: _benchmark_target(target, thermal_images)
        for target in targets
    }
    results: dict[str, dict[str, Any]] = {}
    for target in targets:
        results[target.key] = {
            "display_name": target.display_name,
            "preprocessing": target.preprocessing_name,
            "weights": str(target.weights),
            **metrics[target.key],
            **benchmarks[target.key],
            "confusion_matrix": confusions[target.key],
        }

    output_dir = project_root / "results" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "comparison": results,
        "protocol": {
            "same_checkpoint": True,
            "checkpoint": str(targets[0].weights),
            "training_image_size": FINAL_TRAINING_IMAGE_SIZE,
            "original_inference_image_size": FINAL_TRAINING_IMAGE_SIZE,
            "proposed_inference_image_size": FINAL_INFERENCE_IMAGE_SIZE,
            "original_preprocessing": "legacy percentile normalization + OpenCV CLAHE",
            "proposed_preprocessing": "uint16 histogram percentile normalization + cached OpenCV CLAHE",
            "validation_images": len(thermal_images),
            "batch": 1,
            "warmup": 20,
            "repeats": 3,
            "disk_io_included": False,
            "cuda_synchronized": True,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
    }
    json_path = output_dir / "06_clahe_optimization_640_vs_960.json"
    csv_path = output_dir / "06_clahe_optimization_640_vs_960.csv"
    figure_path = output_dir / "06_clahe_optimization_640_vs_960.png"
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, results)
    _create_figure(figure_path, results, torch.cuda.get_device_name(0))
    print(
        f"\n[COMPLETE]\n  Figure: {figure_path}\n  Metrics: {json_path}\n  Table: {csv_path}"
    )
    return results
