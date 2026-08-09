"""Generate final image and video demos without training or dataset mutation."""

import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.preprocessing.thermal import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE

DEMO_CONF = 0.25
MATCH_IOU = 0.50
DEMO_SEED = 23
IMAGE_SIZE = 640
HEADER_HEIGHT = 72
VIDEO_INPUT_NAME = "test04.mp4"

BACKGROUND = (24, 31, 45)
CARD = (32, 41, 57)
WHITE = (245, 247, 250)
MUTED = (180, 190, 205)
GT_COLOR = (255, 235, 40)
PRED_COLOR = (20, 170, 255)
ACCENT = (30, 210, 145)


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class SampleResult:
    path: Path
    image: np.ndarray
    ground_truth: list[tuple[int, int, int, int]]
    predictions: list[Detection]
    tp: int
    fn: int
    fp: int

    @property
    def recall(self) -> float:
        return self.tp / len(self.ground_truth)


def _model_path(project_root: Path) -> Path:
    current_path = project_root / "results" / "clahe" / "best.pt"
    legacy_path = project_root / "models" / "clahe" / "best.pt"
    for path in (current_path, legacy_path):
        if path.is_file():
            return path
    raise FileNotFoundError(f"CLAHE best.pt not found at {current_path} or {legacy_path}")


def load_model(project_root: Path) -> tuple[Any, Any, Path]:
    """Load the final model once and choose CUDA when available."""
    config_dir = Path(tempfile.gettempdir()) / "ThermalSight_Ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("YOLO_VERBOSE", "false")
    import torch
    from ultralytics import YOLO

    path = _model_path(project_root)
    device = 0 if torch.cuda.is_available() else "cpu"
    return YOLO(str(path)), device, path


def select_validation_samples(project_root: Path, seed: int = DEMO_SEED) -> list[Path]:
    """Select four deterministic positive CLAHE validation images."""
    image_dir = project_root / "data" / "processed" / "thermal_night" / "model_input" / "clahe" / "images" / "val"
    label_dir = project_root / "data" / "processed" / "thermal_night" / "model_input" / "clahe" / "labels" / "val"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError("CLAHE validation images or labels are missing")
    candidates = []
    for image_path in sorted(image_dir.glob("*.png")):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        rows = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(len(row) >= 5 and row[0] == "0" for row in rows):
            candidates.append(image_path)
    if len(candidates) < 4:
        raise RuntimeError(f"Need four positive validation images, found {len(candidates)}")
    return random.Random(seed).sample(candidates, 4)


def yolo_to_xyxy(row: list[str], width: int, height: int) -> tuple[int, int, int, int]:
    _, center_x, center_y, box_width, box_height = row[:5]
    cx, cy, bw, bh = map(float, (center_x, center_y, box_width, box_height))
    x1 = int(round((cx - bw / 2) * width))
    y1 = int(round((cy - bh / 2) * height))
    x2 = int(round((cx + bw / 2) * width))
    y2 = int(round((cy + bh / 2) * height))
    return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)


def load_yolo_labels(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        row = line.split()
        if len(row) >= 5 and row[0] == "0":
            boxes.append(yolo_to_xyxy(row, width, height))
    return boxes


def compute_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def match_detections(
    ground_truth: list[tuple[int, int, int, int]], predictions: list[Detection]
) -> tuple[int, int, int]:
    """Greedily match the highest-IoU pairs one-to-one at IoU >= 0.5."""
    pairs = sorted(
        (
            (compute_iou(gt_box, prediction.box), gt_index, prediction_index)
            for gt_index, gt_box in enumerate(ground_truth)
            for prediction_index, prediction in enumerate(predictions)
        ),
        reverse=True,
    )
    matched_gt: set[int] = set()
    matched_predictions: set[int] = set()
    for iou, gt_index, prediction_index in pairs:
        if iou < MATCH_IOU:
            break
        if gt_index not in matched_gt and prediction_index not in matched_predictions:
            matched_gt.add(gt_index)
            matched_predictions.add(prediction_index)
    tp = len(matched_gt)
    return tp, len(ground_truth) - tp, len(predictions) - tp


def _extract_detections(result: Any) -> list[Detection]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    return [
        Detection(tuple(int(round(value)) for value in box), float(confidence))
        for box, confidence, class_id in zip(xyxy, confidences, classes)
        if class_id == 0
    ]


def _draw_dashed_rectangle(
    image: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int], thickness: int = 1
) -> None:
    x1, y1, x2, y2 = box
    dash = 10
    for start in range(x1, x2, dash * 2):
        cv2.line(image, (start, y1), (min(start + dash, x2), y1), color, thickness, cv2.LINE_AA)
        cv2.line(image, (start, y2), (min(start + dash, x2), y2), color, thickness, cv2.LINE_AA)
    for start in range(y1, y2, dash * 2):
        cv2.line(image, (x1, start), (x1, min(start + dash, y2)), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x2, start), (x2, min(start + dash, y2)), color, thickness, cv2.LINE_AA)


def _rectangles_overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    return not (first[2] <= second[0] or second[2] <= first[0] or first[3] <= second[1] or second[3] <= first[1])


def _draw_predictions(image: np.ndarray, detections: list[Detection]) -> None:
    """Draw boxes and place compact labels without overlapping other labels."""
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(image, (x1, y1), (x2, y2), PRED_COLOR, 3, cv2.LINE_AA)
    occupied: list[tuple[int, int, int, int]] = []
    for detection in sorted(detections, key=lambda item: (item.box[1], item.box[0])):
        x1, y1, x2, _ = detection.box
        label = f"Person {detection.confidence:.2f}"
        scale = 0.40
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        label_width, label_height = text_width + 8, text_height + 9
        x_candidates = (x1, x2 - label_width, x1 - label_width // 2)
        y_candidates = [y1 - label_height - tier * (label_height + 3) for tier in range(5)]
        y_candidates.extend((y1 + 3, y1 + label_height + 6))
        chosen = None
        for candidate_y in y_candidates:
            for candidate_x in x_candidates:
                left = min(max(0, candidate_x), max(0, image.shape[1] - label_width))
                top = min(max(0, candidate_y), max(0, image.shape[0] - label_height))
                rectangle = (left, top, left + label_width, top + label_height)
                if not any(_rectangles_overlap(rectangle, previous) for previous in occupied):
                    chosen = rectangle
                    break
            if chosen is not None:
                break
        if chosen is None:
            left = min(max(0, x1), max(0, image.shape[1] - label_width))
            top = min(max(0, y1 - label_height), max(0, image.shape[0] - label_height))
            chosen = (left, top, left + label_width, top + label_height)
        occupied.append(chosen)
        left, top, right, bottom = chosen
        cv2.rectangle(image, (left, top), (right, bottom), PRED_COLOR, -1)
        cv2.putText(
            image, label, (left + 4, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX,
            scale, (12, 20, 30), 1, cv2.LINE_AA,
        )


def _draw_prediction(image: np.ndarray, detection: Detection) -> None:
    """Backward-compatible single-detection drawing helper."""
    _draw_predictions(image, [detection])


def _put_centered(
    image: np.ndarray, text: str, center_x: int, y: int, scale: float, color: tuple[int, int, int], thickness: int
) -> None:
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    cv2.putText(image, text, (center_x - width // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _fit_without_distortion(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    canvas = np.full((height, width, 3), CARD, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _sample_card(sample: SampleResult, width: int = 640, image_height: int = 512) -> np.ndarray:
    rendered = sample.image.copy()
    for box in sample.ground_truth:
        _draw_dashed_rectangle(rendered, box, GT_COLOR, 2)
    _draw_predictions(rendered, sample.predictions)
    rendered = _fit_without_distortion(rendered, width, image_height)
    card = np.full((image_height + 94, width, 3), CARD, dtype=np.uint8)
    card[44:44 + image_height] = rendered
    name = sample.path.stem if len(sample.path.stem) <= 58 else f"{sample.path.stem[:55]}..."
    cv2.putText(card, name, (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.51, WHITE, 1, cv2.LINE_AA)
    stats = (
        f"GT {len(sample.ground_truth)}  |  TP {sample.tp}  |  FN {sample.fn}  |  "
        f"FP {sample.fp}  |  Recall {sample.recall * 100:.1f}%"
    )
    cv2.putText(card, stats, (14, image_height + 79), cv2.FONT_HERSHEY_SIMPLEX, 0.54, WHITE, 1, cv2.LINE_AA)
    cv2.rectangle(card, (0, 0), (width - 1, card.shape[0] - 1), (60, 74, 95), 1)
    return card


def draw_validation_demo(samples: list[SampleResult], output_path: Path) -> None:
    card_width, card_height, gap, margin = 640, 606, 18, 24
    header_height, summary_height = 105, 72
    canvas_width = margin * 2 + card_width * 2 + gap
    canvas_height = header_height + card_height * 2 + gap + summary_height + margin
    canvas = np.full((canvas_height, canvas_width, 3), BACKGROUND, dtype=np.uint8)
    _put_centered(canvas, "THERMALSIGHT | SELECTED 4-SAMPLE DETECTION DEMO", canvas_width // 2, 39, 0.82, WHITE, 2)
    _put_centered(
        canvas, f"Demo seed {DEMO_SEED}  |  CLAHE + YOLO26n  |  GT dashed  |  Prediction solid  |  IoU >= 0.50",
        canvas_width // 2, 72, 0.54, MUTED, 1,
    )
    for index, sample in enumerate(samples):
        row, column = divmod(index, 2)
        x = margin + column * (card_width + gap)
        y = header_height + row * (card_height + gap)
        canvas[y:y + card_height, x:x + card_width] = _sample_card(sample)
    total_gt = sum(len(sample.ground_truth) for sample in samples)
    total_tp = sum(sample.tp for sample in samples)
    total_fn = sum(sample.fn for sample in samples)
    total_fp = sum(sample.fp for sample in samples)
    recall = total_tp / total_gt
    summary = (
        f"4-Sample Demo Recall: {recall * 100:.1f}%   |   GT Persons {total_gt}   |   "
        f"TP {total_tp}   |   FN {total_fn}   |   FP {total_fp}"
    )
    _put_centered(canvas, summary, canvas_width // 2, canvas_height - 39, 0.64, ACCENT, 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to write image demo: {output_path}")


def run_image_demo(project_root: Path, model_bundle: tuple[Any, Any, Path] | None = None) -> dict[str, Any]:
    selected = select_validation_samples(project_root)
    model, device, model_path = model_bundle or load_model(project_root)
    predictions = model.predict(
        source=[str(path) for path in selected], imgsz=IMAGE_SIZE, conf=DEMO_CONF,
        classes=[0], device=device, verbose=False, save=False,
    )
    label_dir = project_root / "data" / "processed" / "thermal_night" / "model_input" / "clahe" / "labels" / "val"
    samples = []
    for path, prediction_result in zip(selected, predictions):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to read validation image: {path}")
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gt = load_yolo_labels(label_dir / f"{path.stem}.txt", image.shape[1], image.shape[0])
        detections = _extract_detections(prediction_result)
        tp, fn, fp = match_detections(gt, detections)
        samples.append(SampleResult(path, image_bgr, gt, detections, tp, fn, fp))
    output_path = project_root / "results" / "demo" / "val_demo_4samples.png"
    draw_validation_demo(samples, output_path)

    print(f"\n[Validation Demo]\n\nDemo seed:\n{DEMO_SEED}\n\nSelected:")
    for path in selected:
        print(f"- {path.name}")
    print()
    for index, sample in enumerate(samples, 1):
        print(
            f"Sample {index} | GT {len(sample.ground_truth)} | TP {sample.tp} | "
            f"FN {sample.fn} | FP {sample.fp} | Recall {sample.recall * 100:.1f}%"
        )
    total_gt = sum(len(sample.ground_truth) for sample in samples)
    total_tp = sum(sample.tp for sample in samples)
    total_fn = sum(sample.fn for sample in samples)
    total_fp = sum(sample.fp for sample in samples)
    print(f"\n4-Sample Demo Recall: {total_tp / total_gt * 100:.1f}%")
    print(f"\nModel:\n{model_path}\n\nSaved:\n{output_path}")
    return {"samples": samples, "gt": total_gt, "tp": total_tp, "fn": total_fn, "fp": total_fp, "output": output_path}


def apply_clahe_to_video_frame(frame: np.ndarray, clahe: Any) -> np.ndarray:
    """Apply only CLAHE to an already baseline-normalized uint8 MP4 frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return clahe.apply(gray)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    rendered = frame.copy()
    _draw_predictions(rendered, detections)
    count_text = f"Persons: {len(detections)}"
    (width, height), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    x1, y1 = rendered.shape[1] - width - 28, rendered.shape[0] - height - 24
    overlay = rendered.copy()
    cv2.rectangle(overlay, (x1 - 10, y1 - 10), (rendered.shape[1] - 10, rendered.shape[0] - 10), (12, 20, 30), -1)
    cv2.addWeighted(overlay, 0.72, rendered, 0.28, 0, rendered)
    cv2.putText(rendered, count_text, (x1, rendered.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)
    return rendered


def _compose_video_frame(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height, width = left.shape[:2]
    canvas = np.full((height + HEADER_HEIGHT, width * 2, 3), BACKGROUND, dtype=np.uint8)
    canvas[HEADER_HEIGHT:, :width] = left
    canvas[HEADER_HEIGHT:, width:] = right
    _put_centered(canvas, "BASELINE THERMAL", width // 2, 43, 0.72, WHITE, 2)
    _put_centered(canvas, "THERMALSIGHT", width + width // 2, 30, 0.72, WHITE, 2)
    _put_centered(canvas, "CLAHE + YOLO26n", width + width // 2, 55, 0.50, ACCENT, 1)
    cv2.line(canvas, (width, 0), (width, height + HEADER_HEIGHT), (95, 110, 132), 2)
    return canvas


def create_video_demo(project_root: Path, model: Any, device: Any, output_path: Path) -> dict[str, Any]:
    input_path = project_root / "data" / "demo" / VIDEO_INPUT_NAME
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open demo input: {input_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError("Input video metadata is invalid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (width * 2, height + HEADER_HEIGHT),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open output video writer: {output_path}")
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    processed = 0
    try:
        while True:
            ok, source_frame = capture.read()
            if not ok:
                break
            clahe_gray = apply_clahe_to_video_frame(source_frame, clahe)
            clahe_bgr = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
            prediction = model.predict(
                source=clahe_bgr, imgsz=IMAGE_SIZE, conf=DEMO_CONF,
                classes=[0], device=device, verbose=False, save=False,
            )[0]
            right = draw_detections(clahe_bgr, _extract_detections(prediction))
            writer.write(_compose_video_frame(source_frame, right))
            processed += 1
            if processed % 100 == 0 or processed == frame_count:
                print(f"{processed} / {frame_count}")
    finally:
        capture.release()
        writer.release()
    if processed != frame_count:
        raise RuntimeError(f"Processed {processed} frames, expected {frame_count}")
    return {
        "input": input_path, "frames": processed, "width": width * 2,
        "height": height + HEADER_HEIGHT, "fps": fps, "output": output_path,
    }


def _verify_video(path: Path, expected: dict[str, Any]) -> None:
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_ok, _ = capture.read()
    capture.release()
    if not opened or not first_ok or frames <= 0 or fps <= 0:
        raise RuntimeError(f"Output video playback verification failed: {path}")
    if frames != expected["frames"] or width != expected["width"] or height != expected["height"]:
        raise RuntimeError("Output video metadata does not match the generated frames")
    if abs(fps - expected["fps"]) > 0.01:
        raise RuntimeError(f"Output FPS {fps} does not match input FPS {expected['fps']}")


def run_video_demo(project_root: Path, model_bundle: tuple[Any, Any, Path] | None = None) -> dict[str, Any]:
    model, device, model_path = model_bundle or load_model(project_root)
    output_path = project_root / "results" / "demo" / "demo.mp4"
    print(
        f"\n[Video Demo]\n\nInput:\n{project_root / 'data' / 'demo' / VIDEO_INPUT_NAME}"
        f"\n\nModel:\n{model_path}\n\nCLAHE:\nclipLimit={CLAHE_CLIP_LIMIT}, "
        f"tileGridSize={CLAHE_TILE_GRID_SIZE}\n\nConfidence:\n{DEMO_CONF}\n\nProcessing:"
    )
    details = create_video_demo(project_root, model, device, output_path)
    _verify_video(output_path, details)
    print(f"\nSaved:\n{output_path}")
    return details


def run_all_demos(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create both demos while sharing one loaded model instance."""
    model_bundle = load_model(project_root)
    image_details = run_image_demo(project_root, model_bundle)
    video_details = run_video_demo(project_root, model_bundle)
    return image_details, video_details
