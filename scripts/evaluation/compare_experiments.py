"""Compare completed experiments without loading or running a YOLO model."""

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from scripts.preprocessing.dataset_builder import create_preview  # noqa: E402

EXPERIMENTS = ("baseline", "clahe", "bilateral_clahe")
DISPLAY_NAMES = {
    "baseline": "Baseline",
    "clahe": "CLAHE",
    "bilateral_clahe": "Bilateral+CLAHE",
}
METRICS = ("precision", "recall", "map50", "map50_95")
COMMON_CONFIG_KEYS = (
    "model", "pretrained", "epochs", "imgsz", "batch", "seed", "workers",
    "device", "optimizer", "patience", "deterministic", "augmentation",
)


def _load_results(project_root: Path) -> list[dict[str, Any]]:
    latency_path = project_root / "results" / "preprocessing" / "preprocessing_latency.csv"
    with latency_path.open(encoding="utf-8-sig", newline="") as file:
        latency = {row["experiment"]: row for row in csv.DictReader(file)}
    if set(latency) != set(EXPERIMENTS):
        raise ValueError("Latency CSV must contain exactly the three experiments")

    rows: list[dict[str, Any]] = []
    config_reference: dict[str, Any] | None = None
    for experiment in EXPERIMENTS:
        summary_path = project_root / "results" / experiment / "training_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("experiment") != experiment:
            raise ValueError(f"Experiment mismatch in {summary_path}")
        metrics = summary.get("metrics")
        if not isinstance(metrics, dict) or any(key not in metrics for key in METRICS):
            raise ValueError(f"Missing metrics in {summary_path}")
        snapshot = summary.get("common_config_snapshot", {})
        common = {key: snapshot.get(key) for key in COMMON_CONFIG_KEYS}
        if config_reference is None:
            config_reference = common
        elif common != config_reference:
            raise ValueError(f"Common training config differs for {experiment}")
        if summary.get("epochs_completed") != summary.get("epochs_configured"):
            raise ValueError(f"Training did not complete configured epochs: {experiment}")

        best_model = project_root / "results" / experiment / "best.pt"
        if not best_model.is_file():
            raise FileNotFoundError(f"Best model not found: {best_model}")
        latency_row = latency[experiment]
        rows.append({
            "experiment": experiment,
            **{key: float(metrics[key]) for key in METRICS},
            "preprocess_mean_ms": float(latency_row["mean_ms"]),
            "preprocess_median_ms": float(latency_row["median_ms"]),
            "preprocess_p95_ms": float(latency_row["p95_ms"]),
            "best_model_path": str(best_model.resolve()),
        })
    return rows


def _delta(new: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    return {
        **{key: (new[key] - reference[key]) * 100.0 for key in METRICS},
        "latency_ms": new["preprocess_mean_ms"] - reference["preprocess_mean_ms"],
    }


def _write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "experiment", *METRICS, "preprocess_mean_ms", "preprocess_median_ms",
        "preprocess_p95_ms", "best_model_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = row.copy()
            for key in METRICS:
                formatted[key] = f"{row[key]:.6f}"
            for key in ("preprocess_mean_ms", "preprocess_median_ms", "preprocess_p95_ms"):
                formatted[key] = f"{row[key]:.4f}"
            writer.writerow(formatted)


def _plot_metric_delta(path: Path, rows: list[dict[str, Any]]) -> None:
    by_name = {row["experiment"]: row for row in rows}
    baseline = by_name["baseline"]
    comparisons = ("clahe", "bilateral_clahe")
    colors = ("#F58518", "#54A24B")
    x = list(range(len(METRICS)))
    width = 0.34
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for index, (experiment, color) in enumerate(zip(comparisons, colors)):
        positions = [value + (index - 0.5) * width for value in x]
        values = [(by_name[experiment][key] - baseline[key]) * 100.0 for key in METRICS]
        bars = axis.bar(
            positions, values, width,
            label=f"{DISPLAY_NAMES[experiment]} vs Baseline", color=color,
        )
        axis.bar_label(
            bars, labels=[f"{value:+.2f}" for value in values],
            padding=3, fontsize=8,
        )
    axis.set_xticks(x, ["Precision", "Recall", "mAP50", "mAP50-95"])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Change vs Baseline (%p)")
    axis.set_title("Detection Metric Gains and Losses vs Baseline")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_validation_recall_curves(project_root: Path, path: Path) -> None:
    colors = ("#4C78A8", "#F58518", "#54A24B")
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for experiment, color in zip(EXPERIMENTS, colors):
        results_path = project_root / "results" / experiment / "results.csv"
        with results_path.open(encoding="utf-8-sig", newline="") as file:
            epoch_rows = list(csv.DictReader(file))
        if not epoch_rows or "metrics/recall(B)" not in epoch_rows[0]:
            raise ValueError(f"Validation Recall column not found: {results_path}")
        epochs = [int(float(row["epoch"])) for row in epoch_rows]
        recalls = [float(row["metrics/recall(B)"]) for row in epoch_rows]
        axis.plot(
            epochs, recalls, color=color, linewidth=1.8,
            label=DISPLAY_NAMES[experiment],
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation Recall")
    axis.set_title("Validation Recall across Training")
    axis.set_xlim(left=1)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_recall_latency(path: Path, rows: list[dict[str, Any]]) -> None:
    colors = ("#4C78A8", "#F58518", "#54A24B")
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    for row, color in zip(rows, colors):
        x = row["preprocess_mean_ms"]
        y = row["recall"]
        label = DISPLAY_NAMES[row["experiment"]]
        axis.scatter(x, y, s=90, color=color)
        axis.annotate(label, (x, y), xytext=(7, 7), textcoords="offset points", fontsize=9)
    axis.set_xlabel("Preprocessing Mean Latency (ms)")
    axis.set_ylabel("Recall")
    axis.set_title("Recall vs. Preprocessing Latency")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _comparison_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Experiment | Precision | Recall | mAP50 | mAP50-95 | Mean latency (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {DISPLAY_NAMES[row['experiment']]} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['map50']:.4f} | {row['map50_95']:.4f} | "
            f"{row['preprocess_mean_ms']:.3f} |"
        )
    return "\n".join(lines)


def _delta_line(name: str, delta: dict[str, float]) -> str:
    return (
        f"- {name}: Precision {delta['precision']:+.2f}%p, Recall {delta['recall']:+.2f}%p, "
        f"mAP50 {delta['map50']:+.2f}%p, mAP50-95 {delta['map50_95']:+.2f}%p, "
        f"mean latency {delta['latency_ms']:+.3f} ms"
    )


def _write_markdown_outputs(project_root: Path, output_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_name = {row["experiment"]: row for row in rows}
    clahe_delta = _delta(by_name["clahe"], by_name["baseline"])
    bilateral_delta = _delta(by_name["bilateral_clahe"], by_name["baseline"])
    bilateral_vs_clahe = _delta(by_name["bilateral_clahe"], by_name["clahe"])
    winner = max(rows, key=lambda row: row["recall"])
    if winner["experiment"] != "clahe":
        raise RuntimeError("CLAHE is not the highest-Recall experiment in the stored results")
    table = _comparison_table(rows)

    comparison = f"""# ThermalSight Experiment Comparison

## 실험 조건

YOLO26n, 50 epochs, image size 640, batch 16, seed 42를 공통으로 사용했으며 열영상 전처리만 변경했다.

## 결과

{table}

전처리 latency는 현재 개발 PC에서 측정한 순수 전처리 함수의 상대 비교값이다.

## Baseline 대비 변화

{_delta_line('CLAHE vs Baseline', clahe_delta)}
{_delta_line('Bilateral+CLAHE vs Baseline', bilateral_delta)}
{_delta_line('Bilateral+CLAHE vs CLAHE', bilateral_vs_clahe)}

## 최종 선택

CLAHE를 선택한다. 야간 보행자 미탐지 최소화를 위해 Recall을 우선했으며, CLAHE가 본 Validation set에서 가장 높은 Recall을 기록했다. Baseline보다 Precision은 유지·소폭 개선되고 latency 증가는 제한적이지만, mAP50과 mAP50-95는 소폭 감소했다.

최종 모델: `results/clahe/best.pt`

## 한계

Validation은 112장, person bounding box 508개로 규모가 작다. Recall 개선폭도 작으므로 통계적 유의성이나 일반적 우위를 주장할 수 없으며 추가 데이터에서 검증이 필요하다.
"""
    (output_dir / "experiment_comparison.md").write_text(comparison, encoding="utf-8")

    report = f"""# ThermalSight 모델링 결과

## 1. 모델링 목표

동일한 YOLO26n에서 열영상 전처리 방식에 따른 야간 보행자 탐지 성능을 비교했다.

## 2. 실험 조건

- YOLO26n, COCO pretrained weight
- epochs 50, image size 640, batch 16, seed 42
- Train 2,110장, Validation 112장
- 단일 클래스: person
- 세 실험은 전처리만 변경하고 각각 동일 pretrained weight에서 독립 학습

## 3. 비교한 전처리

- Baseline: 1~99 percentile clipping 후 uint8 정규화
- CLAHE: Baseline 정규화 후 CLAHE 적용
- Bilateral + CLAHE: Baseline 정규화 후 bilateral filter와 CLAHE 적용

## 4. 실험 결과

{table}

전처리 latency는 현재 개발 PC에서 측정한 상대 비교값이며 embedded device 성능이 아니다.

## 5. 결과 해석

Baseline은 가장 낮은 전처리 latency와 가장 높은 mAP를 기록했다. CLAHE는 본 Validation set에서 가장 높은 Recall을 기록했으며 Precision도 소폭 높았다. 다만 mAP는 Baseline보다 소폭 낮다. Bilateral+CLAHE는 연산 비용이 가장 높으면서 Recall과 mAP가 감소해 추가 필터의 이점을 확인하지 못했다.

## 6. 최종 선택

최종 전처리는 **CLAHE**, 모델은 `results/clahe/best.pt`이다. 야간 보행자 탐지에서 False Negative 감소를 우선해 Recall을 핵심 지표로 선택했다. 다만 Validation이 112장으로 작고 개선폭도 작으므로 추가 데이터에서 검증이 필요하다.

## 7. MVP 전달 사항

- 전처리: CLAHE
- 모델: `results/clahe/best.pt`
- 입력: High-bit thermal TIFF 또는 thermal image
- 처리: Thermal image → baseline normalization → CLAHE → YOLO26n → Person bounding boxes
- 출력: 탐지 이미지, Person count, confidence, preprocessing time, inference time
"""
    (output_dir / "ThermalSight_모델링결과.md").write_text(report, encoding="utf-8")


def generate_evaluation_outputs(project_root: Path) -> list[dict[str, Any]]:
    """Read completed results and generate reports plus four presentation plots."""
    rows = _load_results(project_root)
    output_dir = project_root / "results" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_results_csv(output_dir / "experiment_results.csv", rows)
    create_preview(
        project_root / "data" / "processed" / "thermal_night",
        output_dir / "01_preprocessing_preview.png",
    )
    _plot_metric_delta(output_dir / "02_metric_delta_vs_baseline.png", rows)
    _plot_validation_recall_curves(
        project_root, output_dir / "03_validation_recall_curve.png"
    )
    _plot_recall_latency(output_dir / "04_recall_latency_tradeoff.png", rows)
    _write_markdown_outputs(project_root, output_dir, rows)
    for obsolete_name in (
        "detection_metrics_comparison.png",
        "recall_latency_tradeoff.png",
    ):
        obsolete_path = output_dir / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()
    print(f"Evaluation outputs created: {output_dir}")
    return rows
