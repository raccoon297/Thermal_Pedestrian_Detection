"""Preview and model-input dataset builders for thermal preprocessing."""

import csv
import json
import random
import shutil
import statistics
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

from .thermal import PREPROCESSORS

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

SPLITS = ("train", "val")
EXPECTED_COUNTS = {"train": 2110, "val": 112}
EXPECTED_SHAPE = (512, 640)


def _read_thermal(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.dtype != np.uint16 or array.shape != EXPECTED_SHAPE:
        raise ValueError(f"Invalid TIFF {path}: expected uint16 {EXPECTED_SHAPE}, received {array.dtype} {array.shape}")
    return array


def _manifest_candidates(manifest_path: Path) -> tuple[list[str], list[str]]:
    positives, negatives = [], []
    with manifest_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["split"] == "train":
                target = positives if row["has_person"].strip().lower() == "true" else negatives
                target.append(Path(row["file_name"]).stem)
    return positives, negatives


def create_preview(dataset_root: Path, output_path: Path, sample_count: int = 6,
                   seed: int = 42) -> list[Path]:
    """Create a reproducible 6x3 comparison preview from training TIFFs."""
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    manifest_path = dataset_root / "metadata" / "night_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    positives, negatives = _manifest_candidates(manifest_path)
    rng = random.Random(seed)
    stems = rng.sample(positives, sample_count - 1) + rng.sample(negatives, 1)
    rng.shuffle(stems)
    paths = [dataset_root / "source_tiff" / "train" / f"{stem}.tiff" for stem in stems]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Selected TIFF not found: {missing[0]}")

    columns = list(PREPROCESSORS.items())
    fig, axes = plt.subplots(sample_count, len(columns), figsize=(15, 3.2 * sample_count))
    for row_index, path in enumerate(paths):
        source = _read_thermal(path)
        for column_index, (name, preprocessor) in enumerate(columns):
            result = preprocessor(source)
            if result.dtype != np.uint8 or result.shape != EXPECTED_SHAPE:
                raise RuntimeError(f"{name} returned {result.dtype} {result.shape}")
            axes[row_index, column_index].imshow(result, cmap="gray", vmin=0, vmax=255)
            axes[row_index, column_index].set_title(
                f"{name.replace('_', ' ').title()}\n{path.stem[:36]}\n"
                f"min={result.min()} max={result.max()} mean={result.mean():.1f}", fontsize=8)
            axes[row_index, column_index].axis("off")
            print(f"{path.name} | {name}: dtype={result.dtype}, shape={result.shape}, "
                  f"min={result.min()}, max={result.max()}, mean={result.mean():.2f}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return paths


def _source_paths(dataset_root: Path, split: str) -> list[Path]:
    paths = sorted((dataset_root / "source_tiff" / split).glob("*.tiff"))
    if len(paths) != EXPECTED_COUNTS[split]:
        raise RuntimeError(f"{split} source count mismatch: expected {EXPECTED_COUNTS[split]}, got {len(paths)}")
    return paths


def _validate_experiment(dataset_root: Path, experiment: str) -> dict:
    root = dataset_root / "model_input" / experiment
    split_results = {}
    for split in SPLITS:
        images = sorted((root / "images" / split).glob("*.png"))
        labels = sorted((root / "labels" / split).glob("*.txt"))
        if len(images) != EXPECTED_COUNTS[split] or len(labels) != EXPECTED_COUNTS[split]:
            raise RuntimeError(f"{experiment}/{split}: image or label count mismatch")
        if {p.stem for p in images} != {p.stem for p in labels}:
            raise RuntimeError(f"{experiment}/{split}: image/label stems differ")
        source_labels = dataset_root / "labels" / split
        positive_labels = 0
        negative_labels = 0
        for label in labels:
            if label.read_bytes() != (source_labels / label.name).read_bytes():
                raise RuntimeError(f"Copied label differs from source: {label}")
            if label.read_text(encoding="utf-8").strip():
                positive_labels += 1
            else:
                negative_labels += 1
        for image_path in images:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if (image is None or image.ndim != 2 or image.dtype != np.uint8
                    or image.shape != EXPECTED_SHAPE):
                raise RuntimeError(f"Invalid output image: {image_path}")
        split_results[split] = {
            "image_count": len(images),
            "label_count": len(labels),
            "positive_labels": positive_labels,
            "negative_labels": negative_labels,
            "image_label_pairing": True,
            "label_equality": True,
            "corrupted_images": 0,
        }
    return split_results


def _latency_statistics(experiment: str, timings_ms: list[float]) -> dict:
    values = np.asarray(timings_ms, dtype=np.float64)
    return {
        "experiment": experiment,
        "image_count": int(values.size),
        "mean_ms": float(values.mean()),
        "median_ms": float(statistics.median(timings_ms)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def _validate_yaml(yaml_path: Path, experiment: str, dataset_root: Path) -> None:
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
    fields: dict[str, str] = {}
    names_person = False
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "0: person":
            names_person = True
        elif ":" in line and not raw_line.startswith((" ", "\t")):
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    expected_train = f"../../data/processed/thermal_night/model_input/{experiment}/images/train"
    expected_val = f"../../data/processed/thermal_night/model_input/{experiment}/images/val"
    if fields.get("train") != expected_train or fields.get("val") != expected_val:
        raise RuntimeError(f"YAML paths are incorrect: {yaml_path}")
    if not names_person or "names" not in fields:
        raise RuntimeError(f"YAML class mapping is incorrect: {yaml_path}")
    for key in ("train", "val"):
        resolved = (yaml_path.parent / fields[key]).resolve()
        expected = (dataset_root / "model_input" / experiment / "images" / key).resolve()
        if resolved != expected or not resolved.is_dir():
            raise RuntimeError(f"YAML {key} path does not resolve correctly: {yaml_path}")


def _write_latency_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "image_count", "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _has_complete_file_counts(dataset_root: Path, experiment: str) -> bool:
    """Return whether an existing experiment has the expected file counts."""
    root = dataset_root / "model_input" / experiment
    return all(
        len(list((root / "images" / split).glob("*.png"))) == EXPECTED_COUNTS[split]
        and len(list((root / "labels" / split).glob("*.txt"))) == EXPECTED_COUNTS[split]
        for split in SPLITS
    )


def build_model_inputs(dataset_root: Path, experiments: list[str], output_dir: Path,
                       yaml_dir: Path) -> dict:
    """Build and strictly validate PNG/label datasets without changing the split."""
    unknown = set(experiments) - set(PREPROCESSORS)
    if unknown:
        raise ValueError(f"Unknown experiment(s): {sorted(unknown)}")
    source_by_split = {split: _source_paths(dataset_root, split) for split in SPLITS}
    latency_rows = []
    experiment_results = {}
    for experiment in experiments:
        preprocessor = PREPROCESSORS[experiment]
        timings_ms: list[float] = []
        reuse_existing = _has_complete_file_counts(dataset_root, experiment)
        print(f"\nBuilding experiment: {experiment}")
        if reuse_existing:
            print("  Expected file counts already exist; validating and measuring without overwriting.")
        for split in SPLITS:
            image_dir = dataset_root / "model_input" / experiment / "images" / split
            label_dir = dataset_root / "model_input" / experiment / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            for index, source_path in enumerate(source_by_split[split], start=1):
                label_source = dataset_root / "labels" / split / f"{source_path.stem}.txt"
                if not label_source.exists():
                    raise FileNotFoundError(f"Label not found: {label_source}")
                source = _read_thermal(source_path)
                start = time.perf_counter()
                result = preprocessor(source)
                timings_ms.append((time.perf_counter() - start) * 1000.0)
                if reuse_existing:
                    if index % 250 == 0 or index == len(source_by_split[split]):
                        print(f"  {split} timing: {index:,}/{len(source_by_split[split]):,}")
                    continue
                destination = image_dir / f"{source_path.stem}.png"
                if not cv2.imwrite(str(destination), result):
                    raise OSError(f"Failed to write image: {destination}")
                shutil.copy2(label_source, label_dir / label_source.name)
                if index % 250 == 0 or index == len(source_by_split[split]):
                    print(f"  {split}: {index:,}/{len(source_by_split[split]):,}")
        split_result = _validate_experiment(dataset_root, experiment)
        latency = _latency_statistics(experiment, timings_ms)
        latency_rows.append(latency)
        yaml_path = yaml_dir / f"{experiment}.yaml"
        _validate_yaml(yaml_path, experiment, dataset_root)
        experiment_results[experiment] = {
            "splits": split_result,
            "image_dtype": "uint8",
            "image_shape": list(EXPECTED_SHAPE),
            "grayscale": True,
            "preprocessing_latency": latency,
            "yaml_path": str(yaml_path.resolve()),
            "validation_result": "PASS",
        }
        print(f"Validated experiment: {experiment}")
    if len(experiments) > 1:
        for split in SPLITS:
            reference = {p.stem for p in (dataset_root / "model_input" / experiments[0] / "images" / split).glob("*.png")}
            for experiment in experiments[1:]:
                stems = {p.stem for p in (dataset_root / "model_input" / experiment / "images" / split).glob("*.png")}
                if stems != reference:
                    raise RuntimeError(f"Experiment image stems differ in {split}")
    latency_path = output_dir / "preprocessing_latency.csv"
    _write_latency_csv(latency_path, latency_rows)
    report = {
        "experiments": experiment_results,
        "same_image_stems_across_experiments": True,
        "same_labels_across_experiments": True,
        "latency_note": "Development PC measurements for relative comparison; not embedded-device latency.",
        "all_checks_passed": True,
    }
    report_path = output_dir / "model_input_build_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nALL CHECKS PASSED")
    print(f"Latency CSV: {latency_path}")
    print(f"Build report: {report_path}")
    return report
