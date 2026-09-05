"""Build the night subset and all three YOLO26n preprocessing datasets.

One-click data preparation:
1. Extract raw FLIR night uint16 TIFFs and person-only YOLO labels.
2. Build Baseline, CLAHE, and Bilateral+CLAHE uint8 model inputs.
3. Validate image/label counts, pairing, formats, and dataset YAML files.

The optimized CLAHE (960) final pipeline is deliberately not generated here.
It is an inference-time operating point evaluated by compare_yolo26m.py and
used by create_demo.py.
"""

from pathlib import Path

from scripts.data.check_processed import build_and_validate_dataset
from scripts.preprocessing.dataset_builder import build_model_inputs
from scripts.preprocessing.thermal import PREPROCESSORS


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "FLIR_ADAS_v2"
MANIFEST_PATH = PROJECT_ROOT / "results" / "eda" / "night_manifest.csv"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "thermal_night"
PREPROCESSING_RESULTS = PROJECT_ROOT / "results" / "preprocessing"
DATASET_CONFIGS = PROJECT_ROOT / "configs" / "datasets"


def main() -> None:
    print("=" * 72)
    print("ThermalSight data preparation")
    print("FLIR night extraction -> Baseline / CLAHE / Bilateral+CLAHE")
    print("=" * 72)

    try:
        print("\n[1/2] Extracting night thermal TIFFs and person labels")
        extraction_ok = build_and_validate_dataset(
            raw_root=RAW_ROOT,
            manifest_path=MANIFEST_PATH,
            output_root=PROCESSED_ROOT,
        )
        if not extraction_ok:
            raise RuntimeError("Night thermal dataset integrity validation failed")

        print("\n[2/2] Building the three preprocessing model-input datasets")
        build_model_inputs(
            dataset_root=PROCESSED_ROOT,
            experiments=list(PREPROCESSORS),
            output_dir=PREPROCESSING_RESULTS,
            yaml_dir=DATASET_CONFIGS,
        )
    except Exception as error:
        print("\n" + "=" * 72)
        print("DATA PREPARATION FAILED")
        print(f"{type(error).__name__}: {error}")
        print("=" * 72)
        raise

    print("\n" + "=" * 72)
    print("DATA PREPARATION COMPLETE")
    print(f"Dataset: {PROCESSED_ROOT}")
    print("Next: run train.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
