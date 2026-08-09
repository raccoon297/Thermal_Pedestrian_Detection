"""Create six baseline-normalized thermal demo videos."""

from pathlib import Path

from scripts.demo.base_video_builder import build_base_videos

PROJECT_ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    build_base_videos(PROJECT_ROOT)
