from pathlib import Path

from scripts.eda.metadata_eda import run_eda
from scripts.eda.visual_check import create_visual_check


# ============================================================
# Project Path
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent


DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "FLIR_ADAS_v2"
)


TRAIN_ROOT = (
    DATA_ROOT
    / "images_thermal_train"
)


VAL_ROOT = (
    DATA_ROOT
    / "images_thermal_val"
)


TRAIN_COCO = (
    TRAIN_ROOT
    / "coco.json"
)


VAL_COCO = (
    VAL_ROOT
    / "coco.json"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "eda"
)


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "=" * 70
    )

    print(
        "ThermalSight - "
        "FLIR ADAS v2 EDA"
    )

    print(
        "=" * 70
    )

    coco_paths = {
        "train": TRAIN_COCO,
        "val": VAL_COCO,
    }

    # --------------------------------------------------------
    # File Check
    # --------------------------------------------------------

    missing_files = [
        path
        for path in coco_paths.values()
        if not path.exists()
    ]

    if missing_files:

        print(
            "\n[ERROR] "
            "COCO annotation 파일을 "
            "찾을 수 없습니다."
        )

        for path in missing_files:
            print(
                f" - {path}"
            )

        return

    # --------------------------------------------------------
    # Metadata EDA
    # --------------------------------------------------------

    run_eda(
        coco_paths=coco_paths,
        output_dir=OUTPUT_DIR,
    )

    # --------------------------------------------------------
    # Actual Thermal Image Check
    # --------------------------------------------------------

    create_visual_check(
        coco_path=TRAIN_COCO,
        split_root=TRAIN_ROOT,
        save_path=(
            OUTPUT_DIR
            / "01_night_sample_with_bbox.png"
        ),
        sample_count=6,
        seed=42,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EDA 완료"
    )

    print(
        f"결과 위치: "
        f"{OUTPUT_DIR}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
