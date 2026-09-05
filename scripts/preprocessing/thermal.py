"""Thermal preprocessing for the ablation study and final deployment pipeline.

The three public experiment preprocessors intentionally retain the original
implementation used to create checkpoint-1 training data.  The optimized
CLAHE path is separate so the historical experiment remains reproducible.
"""

from collections.abc import Callable
from math import ceil, floor
from threading import local

import cv2
import numpy as np

LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
BILATERAL_DIAMETER = 5
BILATERAL_SIGMA_COLOR = 50.0
BILATERAL_SIGMA_SPACE = 50.0
_CLAHE_LOCAL = local()


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


def _histogram_percentile_bounds(
    image: np.ndarray, lower_percentile: float, upper_percentile: float
) -> tuple[float, float]:
    """Return two NumPy-compatible linear percentiles from one histogram."""
    histogram = cv2.calcHist([image], [0], None, [65536], [0, 65536]).reshape(-1)
    cumulative = np.cumsum(histogram, dtype=np.int64)

    def linear_percentile(percentile: float) -> float:
        rank = (image.size - 1) * (percentile / 100.0)
        lower_rank = floor(rank)
        upper_rank = ceil(rank)
        lower_value = int(np.searchsorted(cumulative, lower_rank + 1))
        upper_value = int(np.searchsorted(cumulative, upper_rank + 1))
        fraction = rank - lower_rank
        return lower_value + (upper_value - lower_value) * fraction

    return linear_percentile(lower_percentile), linear_percentile(upper_percentile)


def _normalize_uint16_histogram(
    image: np.ndarray,
    lower_percentile: float = LOWER_PERCENTILE,
    upper_percentile: float = UPPER_PERCENTILE,
) -> np.ndarray:
    """Fast uint16-to-uint8 percentile normalization for the final pipeline."""
    _validate_image(image)
    if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
    lower, upper = _histogram_percentile_bounds(
        image, lower_percentile, upper_percentile
    )
    if upper <= lower:
        return np.zeros(image.shape, dtype=np.uint8)
    working = image.astype(np.float32)
    np.clip(working, lower, upper, out=working)
    working -= lower
    working *= 255.0 / (upper - lower)
    return np.rint(working).astype(np.uint8)


def _apply_cached_clahe(image: np.ndarray) -> np.ndarray:
    """Reuse one CLAHE instance per thread to avoid per-frame construction."""
    clahe = getattr(_CLAHE_LOCAL, "instance", None)
    if clahe is None:
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID_SIZE,
        )
        _CLAHE_LOCAL.instance = clahe
    return clahe.apply(image)


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


def preprocess_clahe_optimized(image: np.ndarray) -> np.ndarray:
    """Final-model CLAHE path with histogram normalization and CLAHE reuse.

    This produces a practically equivalent image to ``preprocess_clahe`` but
    reduces live raw-frame preprocessing latency.  It is not used to rewrite
    the original ablation datasets.
    """
    return _apply_cached_clahe(_normalize_uint16_histogram(image))


PREPROCESSORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "baseline": preprocess_baseline,
    "clahe": preprocess_clahe,
    "bilateral_clahe": preprocess_bilateral_clahe,
}
