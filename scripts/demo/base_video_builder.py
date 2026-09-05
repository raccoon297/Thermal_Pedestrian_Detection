"""Build deterministic baseline-normalized MP4 files from raw thermal TIFF sequences."""

import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from scripts.preprocessing.thermal import preprocess_baseline

FPS = 10.0
SEQUENCE_COUNT = 6
FRAME_PATTERN = re.compile(r"^(video-.+?)-frame-(\d+)-.*\.(?:tif|tiff)$", re.IGNORECASE)


def find_sequences(source_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """Group TIFF files by video identifier and sort each group by frame number."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Thermal video TIFF directory not found: {source_dir}")
    sequences: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        match = FRAME_PATTERN.match(path.name)
        if match:
            sequences[match.group(1)].append((int(match.group(2)), path))
    for frames in sequences.values():
        frames.sort(key=lambda item: item[0])
    return dict(sequences)


def normalize_frame(image: np.ndarray, path: Path) -> np.ndarray:
    """Apply ThermalSight baseline percentile normalization to one grayscale TIFF."""
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale TIFF, received {image.shape}: {path}")
    if image.dtype == np.uint8:
        image = image.astype(np.uint16)
    elif image.dtype != np.uint16:
        raise TypeError(f"Unsupported TIFF dtype {image.dtype}: {path}")
    return preprocess_baseline(image)


def write_video(output_path: Path, frames: list[tuple[int, Path]]) -> tuple[int, int]:
    """Write one ordered TIFF sequence to an MP4, replacing the destination atomically."""
    if not frames:
        raise ValueError(f"Cannot create an empty video: {output_path}")
    first = cv2.imread(str(frames[0][1]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f"Could not read TIFF: {frames[0][1]}")
    first_gray = normalize_frame(first, frames[0][1])
    height, width = first_gray.shape
    temporary_path = output_path.with_name(f".{output_path.stem}.partial.mp4")
    writer = cv2.VideoWriter(
        str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height), True
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {temporary_path}")
    try:
        total = len(frames)
        for index, (_, path) in enumerate(frames, start=1):
            thermal = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if thermal is None:
                raise RuntimeError(f"Could not read TIFF: {path}")
            gray = normalize_frame(thermal, path)
            if gray.shape != (height, width):
                raise ValueError(f"Inconsistent resolution in {path}: {gray.shape}")
            writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            if index == 1 or index % 100 == 0 or index == total:
                print(f"[{output_path.stem}] {index} / {total}")
    except Exception:
        writer.release()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.release()
    os.replace(temporary_path, output_path)
    return width, height


def verify_video(path: Path) -> dict[str, float | int | bool]:
    """Verify that OpenCV can reopen and decode the generated MP4."""
    capture = cv2.VideoCapture(str(path))
    try:
        opened = capture.isOpened()
        values: dict[str, float | int | bool] = {
            "opened": opened,
            "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        }
        first_frame_ok, _ = capture.read() if opened else (False, None)
        values["first_frame_ok"] = first_frame_ok
        if not opened or not first_frame_ok:
            raise RuntimeError(f"Playback verification failed: {path}")
        if any(float(values[key]) <= 0 for key in ("frames", "width", "height", "fps")):
            raise RuntimeError(f"Invalid video metadata: {path}")
        return values
    finally:
        capture.release()


def build_base_videos(project_root: Path) -> list[Path]:
    """Create test01.mp4 through test06.mp4 from the six longest source sequences."""
    source_dir = project_root / "data" / "raw" / "FLIR_ADAS_v2" / "video_thermal_test"
    output_dir = project_root / "data" / "demo"
    sequences = find_sequences(source_dir)
    if len(sequences) < SEQUENCE_COUNT:
        raise RuntimeError(f"Need at least {SEQUENCE_COUNT} sequences; found {len(sequences)}")

    selected = sorted(sequences.items(), key=lambda item: (-len(item[1]), item[0]))[:SEQUENCE_COUNT]
    print(f"Found {len(sequences)} video sequences.\n")
    print("Selected:")
    for index, (video_id, frames) in enumerate(selected, start=1):
        print(f"test{index:02d} <- {video_id} ({len(frames)} frames)")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[Path, str, dict[str, float | int | bool]]] = []
    for index, (video_id, frames) in enumerate(selected, start=1):
        output_path = output_dir / f"test{index:02d}.mp4"
        width, height = write_video(output_path, frames)
        metadata = verify_video(output_path)
        if metadata["frames"] != len(frames):
            raise RuntimeError(
                f"Frame count mismatch for {output_path}: {metadata['frames']} != {len(frames)}"
            )
        if metadata["width"] != width or metadata["height"] != height:
            raise RuntimeError(f"Resolution mismatch for {output_path}")
        if abs(float(metadata["fps"]) - FPS) > 0.01:
            raise RuntimeError(f"FPS mismatch for {output_path}: {metadata['fps']} != {FPS}")
        results.append((output_path, video_id, metadata))

    print("\nPlayback checks:")
    for path, video_id, metadata in results:
        print(
            f"{path.name}: {video_id}, {metadata['frames']} frames, "
            f"{metadata['width']}x{metadata['height']}, {metadata['fps']:g} fps, PASS"
        )
    return [path for path, _, _ in results]
