"""Preprocessing methods used by the three ThermalSight experiments."""

from collections.abc import Callable

import cv2
import numpy as np

LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
BILATERAL_DIAMETER = 5
BILATERAL_SIGMA_COLOR = 50.0
BILATERAL_SIGMA_SPACE = 50.0


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D grayscale; received shape {image.shape}")
    if image.dtype != np.uint16:
        raise TypeError(f"image dtype must be uint16; received {image.dtype}")
    if image.size == 0:
        raise ValueError("image must not be empty")


def preprocess_baseline(image: np.ndarray, lower_percentile: float = LOWER_PERCENTILE,
                        upper_percentile: float = UPPER_PERCENTILE) -> np.ndarray:
    """Percentile-clip a uint16 thermal frame and normalize it to uint8."""
    _validate_image(image)
    if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
    working = image.astype(np.float32, copy=True)
    if not np.isfinite(working).all():
        raise ValueError("image contains NaN or infinite values")
    lower, upper = np.percentile(working, (lower_percentile, upper_percentile))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("calculated percentile is not finite")
    if upper <= lower:
        return np.zeros(image.shape, dtype=np.uint8)
    clipped = np.clip(working, lower, upper)
    normalized = (clipped - lower) * (255.0 / (upper - lower))
    return np.rint(normalized).astype(np.uint8)


def _apply_clahe(image: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE).apply(image)


def preprocess_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE after the shared percentile-based baseline normalization."""
    return _apply_clahe(preprocess_baseline(image))


def preprocess_bilateral_clahe(image: np.ndarray) -> np.ndarray:
    """Apply edge-preserving bilateral filtering and CLAHE after normalization."""
    baseline = preprocess_baseline(image)
    filtered = cv2.bilateralFilter(baseline, d=BILATERAL_DIAMETER,
                                   sigmaColor=BILATERAL_SIGMA_COLOR,
                                   sigmaSpace=BILATERAL_SIGMA_SPACE)
    return _apply_clahe(filtered)


PREPROCESSORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "baseline": preprocess_baseline,
    "clahe": preprocess_clahe,
    "bilateral_clahe": preprocess_bilateral_clahe,
}
