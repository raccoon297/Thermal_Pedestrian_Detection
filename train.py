"""Train the three YOLO26n (640) preprocessing experiments."""

import argparse

from scripts.model.trainer import EXPERIMENTS, is_training_complete, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one ThermalSight preprocessing experiment")
    parser.add_argument(
        "--experiment",
        choices=EXPERIMENTS,
        help="train one experiment; omit to run all experiments in order",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = (args.experiment,) if args.experiment else EXPERIMENTS

    print("Training order: " + " -> ".join(experiments))
    for experiment in experiments:
        if is_training_complete(experiment):
            print(f"\n[SKIP] {experiment}: essential training artifacts already exist.")
            continue
        print(f"\n[START] {experiment}")
        run_training(experiment)
        print(f"[DONE] {experiment}")


if __name__ == "__main__":
    main()
