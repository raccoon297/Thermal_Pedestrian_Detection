"""Create evaluation figures from the stored experiment results."""

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

EXPERIMENTS = ("baseline", "clahe", "bilateral_clahe")
DISPLAY_NAMES = {
    "baseline": "Baseline",
    "clahe": "CLAHE",
    "bilateral_clahe": "Bilateral + CLAHE",
}
COLORS = {
    "baseline": "#2563EB",
    "clahe": "#F59E0B",
    "bilateral_clahe": "#10B981",
}
METRICS = ("precision", "recall", "map50", "map50_95")
METRIC_LABELS = ("Precision", "Recall", "mAP50", "mAP50-95")
BACKGROUND = "#F8FAFC"
TEXT = "#0F172A"
GRID = "#CBD5E1"
FIGURE_DPI = 240


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _load_inputs(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    latency_path = _require_file(
        project_root / "results" / "preprocessing" / "preprocessing_latency.csv",
        "preprocessing latency CSV",
    )
    with latency_path.open(encoding="utf-8-sig", newline="") as file:
        latency = {row["experiment"]: row for row in csv.DictReader(file)}

    summaries: list[dict[str, Any]] = []
    recall_curves: dict[str, list[float]] = {}
    for experiment in EXPERIMENTS:
        summary_path = _require_file(
            project_root / "results" / experiment / "training_summary.json",
            f"{experiment} training summary",
        )
        results_path = _require_file(
            project_root / "results" / experiment / "results.csv",
            f"{experiment} epoch results",
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary.get("metrics")
        if summary.get("experiment") != experiment or not isinstance(metrics, dict):
            raise ValueError(f"Invalid training summary: {summary_path}")
        if any(key not in metrics for key in METRICS):
            raise ValueError(f"Missing detection metrics: {summary_path}")
        if experiment not in latency:
            raise ValueError(f"Missing latency row for {experiment}: {latency_path}")

        summaries.append({
            "experiment": experiment,
            **{key: float(metrics[key]) for key in METRICS},
            "mean_ms": float(latency[experiment]["mean_ms"]),
        })
        with results_path.open(encoding="utf-8-sig", newline="") as file:
            epoch_rows = list(csv.DictReader(file))
        recall_key = next(
            (key for key in ("metrics/recall(B)", "metrics/recall") if epoch_rows and key in epoch_rows[0]),
            None,
        )
        if recall_key is None:
            raise ValueError(f"Validation Recall column not found: {results_path}")
        recall_curves[experiment] = [float(row[recall_key]) for row in epoch_rows]
    return summaries, recall_curves


def _style_axis(axis: Any) -> None:
    axis.set_facecolor("white")
    axis.tick_params(colors="#334155", labelsize=9)
    for spine in axis.spines.values():
        spine.set_color("#E2E8F0")


def _save(fig: Any, path: Path, dpi: int = FIGURE_DPI) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _select_preview_stems(project_root: Path, sample_count: int = 4) -> list[str]:
    image_sets = []
    for experiment in EXPERIMENTS:
        image_dir = (
            project_root / "data" / "processed" / "thermal_night" / "model_input"
            / experiment / "images" / "train"
        )
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing processed image directory: {image_dir}")
        image_sets.append({path.stem for path in image_dir.glob("*.png")})
    common = sorted(set.intersection(*image_sets))
    if len(common) < sample_count:
        raise RuntimeError(f"Need {sample_count} common processed images, found {len(common)}")
    positions = np.linspace(0, len(common) - 1, sample_count, dtype=int)
    return [common[index] for index in positions]


def _plot_preprocessing_preview(project_root: Path, path: Path) -> None:
    stems = _select_preview_stems(project_root)
    fig, axes = plt.subplots(4, 3, figsize=(13.5, 12.7), facecolor=BACKGROUND)
    fig.suptitle(
        "Comparison of Thermal Preprocessing Methods",
        fontsize=22, fontweight="bold", color=TEXT, y=0.995,
    )
    fig.text(
        0.5, 0.958,
        "The same four FLIR night frames processed with Baseline, CLAHE, and Bilateral + CLAHE",
        ha="center", fontsize=10.5, color="#64748B",
    )
    for column, experiment in enumerate(EXPERIMENTS):
        axes[0, column].set_title(
            DISPLAY_NAMES[experiment], fontsize=14, fontweight="bold",
            color=COLORS[experiment], pad=12,
        )
    for row, stem in enumerate(stems):
        for column, experiment in enumerate(EXPERIMENTS):
            image_path = _require_file(
                project_root / "data" / "processed" / "thermal_night" / "model_input"
                / experiment / "images" / "train" / f"{stem}.png",
                f"{experiment} preview image",
            )
            with Image.open(image_path) as image:
                array = np.asarray(image)
            axis = axes[row, column]
            axis.imshow(array, cmap="gray", vmin=0, vmax=255)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(2.2 if experiment == "clahe" else 1.0)
                spine.set_color(COLORS[experiment] if experiment == "clahe" else "#CBD5E1")
            axis.text(
                0.02, 0.04, f"Mean {array.mean():.1f}", transform=axis.transAxes,
                color="white", fontsize=8, fontweight="bold",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
            )
        axes[row, 0].text(
            -0.035, 0.5, f"S{row + 1}", transform=axes[row, 0].transAxes,
            ha="right", va="center", fontsize=10, fontweight="bold", color="#475569",
        )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.025, wspace=0.035, hspace=0.09)
    _save(fig, path, dpi=180)


def _plot_metric_delta(path: Path, rows: list[dict[str, Any]]) -> None:
    by_name = {row["experiment"]: row for row in rows}
    baseline = by_name["baseline"]
    fig, axis = plt.subplots(figsize=(11, 6.2), facecolor=BACKGROUND)
    _style_axis(axis)
    y = np.arange(len(METRICS))
    height = 0.32
    for offset, experiment in ((-height / 2, "clahe"), (height / 2, "bilateral_clahe")):
        values = np.array([(by_name[experiment][key] - baseline[key]) * 100 for key in METRICS])
        bars = axis.barh(
            y + offset, values, height=height, color=COLORS[experiment],
            label=DISPLAY_NAMES[experiment], alpha=0.92,
        )
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:+.2f}%p", (value, bar.get_y() + bar.get_height() / 2),
                xytext=(6 if value >= 0 else -6, 0), textcoords="offset points",
                va="center", ha="left" if value >= 0 else "right",
                fontsize=9, fontweight="bold", color=TEXT,
            )
    limit = max(
        1.0,
        max(abs((row[key] - baseline[key]) * 100) for row in rows[1:] for key in METRICS) * 1.28,
    )
    axis.set_xlim(-limit, limit)
    axis.set_yticks(y, METRIC_LABELS)
    axis.invert_yaxis()
    axis.axvline(0, color="#334155", linewidth=1.3)
    axis.grid(axis="x", color=GRID, alpha=0.45, linewidth=0.8)
    axis.set_xlabel("Change from Baseline (percentage points)", color=TEXT, labelpad=10)
    axis.set_title("Detection Metric Change from Baseline", loc="left", fontsize=18, fontweight="bold", color=TEXT, pad=18)
    axis.text(
        0, 1.01, "Difference in percentage points; positive values indicate a higher score.",
        transform=axis.transAxes, fontsize=10, color="#64748B", va="bottom",
    )
    axis.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout(pad=2)
    _save(fig, path)


def _plot_recall_curve(path: Path, curves: dict[str, list[float]]) -> None:
    fig, axis = plt.subplots(figsize=(11, 6.2), facecolor=BACKGROUND)
    _style_axis(axis)
    all_values = [value for values in curves.values() for value in values]
    for experiment in EXPERIMENTS:
        recalls = curves[experiment]
        epochs = np.arange(1, len(recalls) + 1)
        axis.plot(
            epochs, recalls, color=COLORS[experiment], linewidth=2.2,
            marker="o", markevery=5, markersize=4.2, markeredgecolor="white",
            label=DISPLAY_NAMES[experiment],
        )
        best_index = int(np.argmax(recalls))
        axis.scatter(
            epochs[best_index], recalls[best_index], s=70,
            color=COLORS[experiment], edgecolor="white", linewidth=1.2, zorder=5,
        )
        axis.annotate(
            f"max {recalls[best_index]:.3f}",
            (epochs[best_index], recalls[best_index]), xytext=(5, 7),
            textcoords="offset points", fontsize=8, color=COLORS[experiment], fontweight="bold",
        )
    lower = max(0.0, min(all_values) - 0.06)
    upper = min(1.0, max(all_values) + 0.05)
    axis.set_ylim(lower, upper)
    axis.set_xlim(1, max(len(values) for values in curves.values()))
    axis.grid(color=GRID, alpha=0.42, linewidth=0.8)
    axis.set_xlabel("Epoch", color=TEXT, labelpad=9)
    axis.set_ylabel("Validation Recall", color=TEXT, labelpad=9)
    axis.set_title("Validation Recall by Training Epoch", loc="left", fontsize=18, fontweight="bold", color=TEXT, pad=18)
    axis.text(
        0, 1.01, "Epoch-level values from results.csv; no smoothing applied.",
        transform=axis.transAxes, fontsize=10, color="#64748B", va="bottom",
    )
    axis.legend(loc="lower right", frameon=False)
    fig.tight_layout(pad=2)
    _save(fig, path)


def _plot_recall_latency(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(10, 6.4), facecolor=BACKGROUND)
    _style_axis(axis)
    x_values = [row["mean_ms"] for row in rows]
    for row in rows:
        experiment = row["experiment"]
        selected = experiment == "clahe"
        axis.scatter(
            row["mean_ms"], row["recall"],
            s=260 if selected else 150, color=COLORS[experiment],
            marker="*" if selected else "o", edgecolor="#92400E" if selected else "white",
            linewidth=0.5, zorder=5,
        )
        axis.annotate(
            f"{DISPLAY_NAMES[experiment]}\nRecall {row['recall']:.3f}  |  {row['mean_ms']:.3f} ms",
            (row["mean_ms"], row["recall"]),
            xytext=(12, 12 if experiment != "bilateral_clahe" else -34),
            textcoords="offset points", fontsize=9, fontweight="bold" if selected else "normal",
            color=TEXT,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92, "edgecolor": COLORS[experiment]},
        )
    margin = max(0.06, (max(x_values) - min(x_values)) * 0.18)
    axis.set_xlim(min(x_values) - margin, max(x_values) + margin)
    axis.set_ylim(0.70, 0.80)
    axis.grid(color=GRID, alpha=0.45, linewidth=0.8)
    axis.set_xlabel("Preprocessing Mean Latency (ms, development PC)", color=TEXT, labelpad=10)
    axis.set_ylabel("Recall", color=TEXT, labelpad=10)
    axis.set_title("Recall and Preprocessing Latency", loc="left", fontsize=19, fontweight="bold", color=TEXT, pad=18)
    axis.text(
        0, 1.01, "Validation recall and mean preprocessing latency measured on the development PC.",
        transform=axis.transAxes, fontsize=10, color="#64748B", va="bottom",
    )
    axis.text(
        0.02, 0.95, "Higher recall / lower latency  ↖", transform=axis.transAxes,
        fontsize=9, color="#64748B", va="top",
    )
    fig.tight_layout(pad=2)
    _save(fig, path)


def _load_confusion_matrices(project_root: Path) -> tuple[dict[str, np.ndarray], float, float]:
    matrices: dict[str, np.ndarray] = {}
    confidence_values: set[float] = set()
    iou_values: set[float] = set()
    label_indexes = {"person": 0, "background": 1}
    for experiment in EXPERIMENTS:
        csv_path = _require_file(
            project_root / "results" / experiment / "confusion_matrix.csv",
            f"{experiment} confusion matrix counts",
        )
        matrix = np.zeros((2, 2), dtype=int)
        with csv_path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != 4:
            raise ValueError(f"Expected four confusion-matrix cells: {csv_path}")
        for row in rows:
            matrix[label_indexes[row["predicted"]], label_indexes[row["true"]]] = int(row["count"])
            confidence_values.add(float(row["confidence_threshold"]))
            iou_values.add(float(row["iou_threshold"]))
        matrices[experiment] = matrix
    if len(confidence_values) != 1 or len(iou_values) != 1:
        raise ValueError("Confusion matrices must use identical confidence and IoU thresholds")
    return matrices, confidence_values.pop(), iou_values.pop()


def _plot_confusion_matrices(project_root: Path, path: Path) -> None:
    matrices, confidence, iou = _load_confusion_matrices(project_root)
    maximum = max(int(matrix.max()) for matrix in matrices.values())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), facecolor=BACKGROUND)
    fig.suptitle(
        "Confusion Matrices by Preprocessing Method",
        fontsize=21, fontweight="bold", color=TEXT, y=0.99,
    )
    fig.text(
        0.5, 0.885,
        f"Validation set  |  Confidence >= {confidence:.2f}  |  IoU > {iou:.2f}  |  Detection counts",
        ha="center", fontsize=10.5, color="#64748B",
    )
    cell_names = np.array([["TP", "FP"], ["FN", "Not applicable"]])
    image = None
    for axis, experiment in zip(axes, EXPERIMENTS):
        matrix = matrices[experiment]
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum, interpolation="nearest")
        axis.set_xticks((0, 1), ("Person", "Background"))
        axis.set_yticks((0, 1), ("Person", "Background"))
        axis.set_xlabel("True class", fontsize=10, fontweight="bold", color="#475569", labelpad=9)
        axis.set_ylabel("Predicted class", fontsize=10, fontweight="bold", color="#475569", labelpad=9)
        axis.set_title(
            DISPLAY_NAMES[experiment], fontsize=15, fontweight="bold",
            color=COLORS[experiment], pad=15,
        )
        axis.tick_params(colors="#334155", labelsize=9, length=0)
        for row in range(2):
            for column in range(2):
                value = matrix[row, column]
                text_color = "white" if value > maximum * 0.48 else TEXT
                if (row, column) == (1, 1):
                    axis.text(
                        column, row, "Not\napplicable",
                        ha="center", va="center", fontsize=11,
                        fontweight="bold", color=text_color, linespacing=1.25,
                    )
                    continue
                axis.text(
                    column, row - 0.08, cell_names[row, column],
                    ha="center", va="center", fontsize=9, fontweight="bold", color=text_color,
                )
                axis.text(
                    column, row + 0.13, f"{value:,}",
                    ha="center", va="center", fontsize=18, fontweight="bold", color=text_color,
                )
        for spine in axis.spines.values():
            spine.set_linewidth(3.0 if experiment == "clahe" else 1.6)
            spine.set_color(COLORS[experiment])
        tp, fp, fn = matrix[0, 0], matrix[0, 1], matrix[1, 0]
        axis.text(
            0.5, -0.23, f"TP {tp:,}   |   FP {fp:,}   |   FN {fn:,}",
            transform=axis.transAxes, ha="center", fontsize=9.5,
            fontweight="bold" if experiment == "clahe" else "normal", color=TEXT,
        )
    fig.subplots_adjust(left=0.055, right=0.89, top=0.79, bottom=0.18, wspace=0.38)
    if image is not None:
        colorbar_axis = fig.add_axes((0.92, 0.20, 0.014, 0.64))
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Number of detections", color="#475569", labelpad=10)
        colorbar.outline.set_edgecolor("#CBD5E1")
    _save(fig, path)


def generate_figures(project_root: Path) -> list[Path]:
    """Generate five figures using stored images and experiment result files."""
    rows, curves = _load_inputs(project_root)
    output_dir = project_root / "results" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "01_preprocessing_preview.png",
        output_dir / "02_metric_delta_vs_baseline.png",
        output_dir / "03_validation_recall_curve.png",
        output_dir / "04_recall_latency_tradeoff.png",
        output_dir / "05_confusion_matrices.png",
    ]
    _plot_preprocessing_preview(project_root, paths[0])
    _plot_metric_delta(paths[1], rows)
    _plot_recall_curve(paths[2], curves)
    _plot_recall_latency(paths[3], rows)
    _plot_confusion_matrices(project_root, paths[4])
    for path in paths:
        print(f"Created: {path}")
    return paths
