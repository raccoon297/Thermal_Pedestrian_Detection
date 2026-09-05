"""Single source of truth for the selected ThermalSight deployment model."""

from pathlib import Path


FINAL_MODEL_NAME = "YOLO26n"
FINAL_PREPROCESSING = "optimized_clahe"
FINAL_TRAINING_IMAGE_SIZE = 640
FINAL_INFERENCE_IMAGE_SIZE = 960
FINAL_CONFIDENCE = 0.25
FINAL_PERSON_CLASS_ID = 0

REFERENCE_MODEL_NAME = "YOLO26m"
REFERENCE_INFERENCE_IMAGE_SIZE = 640


def final_weights_path(project_root: Path) -> Path:
    """Return the trained CLAHE YOLO26n checkpoint selected by the project."""
    return project_root / "results" / "clahe" / "best.pt"
