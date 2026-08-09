"""Generate ThermalSight figures from existing experiment result files."""

from pathlib import Path

from scripts.evaluation.confusion_data import ensure_confusion_data
from scripts.evaluation.visualize_experiments import generate_figures

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    ensure_confusion_data(PROJECT_ROOT)
    generate_figures(PROJECT_ROOT)


if __name__ == "__main__":
    main()
