"""Create figure 06: CLAHE (640) vs optimized CLAHE (960)."""

from pathlib import Path

from scripts.evaluation.clahe_optimization import run_optimization_comparison


PROJECT_ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    run_optimization_comparison(PROJECT_ROOT)
