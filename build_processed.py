from pathlib import Path

from scripts.data.check_processed import (
    build_and_validate_dataset,
)


# ============================================================
# Project Path
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent


RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "FLIR_ADAS_v2"
)


MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "eda"
    / "night_manifest.csv"
)


PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "thermal_night"
)


# ============================================================
# Main
# ============================================================

def main() -> None:

    try:

        success = (
            build_and_validate_dataset(

                raw_root=RAW_ROOT,

                manifest_path=(
                    MANIFEST_PATH
                ),

                output_root=(
                    PROCESSED_ROOT
                ),
            )
        )

    except Exception as error:

        print(
            "\n" + "=" * 70
        )

        print(
            "BUILD FAILED"
        )

        print(
            "=" * 70
        )

        print(
            type(error).__name__
        )

        print(
            error
        )

        return

    if success:

        print(
            "\n다음 단계로 진행 가능합니다."
        )

    else:

        print(
            "\n무결성 검사 실패."
        )

        print(
            "processed dataset을 "
            "학습에 사용하지 마세요."
        )


if __name__ == "__main__":
    main()
