"""Build figure 07: optimized CLAHE + YOLO26n (960) vs YOLO26m (640)."""

from pathlib import Path

from scripts.evaluation.yolo26m_comparison import run_comparison


PROJECT_ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    run_comparison(PROJECT_ROOT)
