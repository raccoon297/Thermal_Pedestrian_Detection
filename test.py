"""Create the final ThermalSight image and video demos."""

import argparse
from pathlib import Path

from scripts.demo.demo_runner import run_all_demos, run_image_demo, run_video_demo

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final ThermalSight demos.")
    parser.add_argument(
        "--mode",
        choices=("images", "video", "all"),
        default="all",
        help="Demo output to create (default: all).",
    )
    args = parser.parse_args()

    if args.mode == "all":
        run_all_demos(PROJECT_ROOT)
    elif args.mode == "images":
        run_image_demo(PROJECT_ROOT)
    else:
        run_video_demo(PROJECT_ROOT)


if __name__ == "__main__":
    main()
