"""Command-line entry point for ThermalSight model-input preparation."""

import argparse
from pathlib import Path

from scripts.preprocessing.dataset_builder import build_model_inputs, create_preview
from scripts.preprocessing.thermal import PREPROCESSORS

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "thermal_night"
PREVIEW_PATH = PROJECT_ROOT / "results" / "preprocessing" / "preprocessing_preview.png"
PREPROCESSING_OUTPUT_DIR = PROJECT_ROOT / "results" / "preprocessing"
YAML_DIR = PROJECT_ROOT / "configs" / "datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ThermalSight model inputs (default: build all experiments)"
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preview", action="store_true", help="create a six-frame comparison")
    action.add_argument("--build", action="store_true", help="build full model-input datasets")
    parser.add_argument("--experiment", choices=tuple(PREPROCESSORS), help="build one experiment only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preview:
        if args.experiment:
            raise SystemExit("--experiment is only valid with --build")
        selected = create_preview(DATASET_ROOT, PREVIEW_PATH)
        print(f"\nPreview created: {PREVIEW_PATH}")
        print(f"Selected images: {len(selected)}")
        return
    experiments = [args.experiment] if args.experiment else list(PREPROCESSORS)
    if not args.build:
        print("No option supplied: building all model-input experiments.")
    build_model_inputs(DATASET_ROOT, experiments, PREPROCESSING_OUTPUT_DIR, YAML_DIR)


if __name__ == "__main__":
    main()
