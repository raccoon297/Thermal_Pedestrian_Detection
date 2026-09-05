import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


# ============================================================
# Constant
# ============================================================

PERSON_CLASS_NAME = "person"


# ============================================================
# TIFF Index
# ============================================================

def build_tiff_index(
    split_root: Path,
) -> dict[str, Path]:
    """
    FLIR split 폴더 내부의 TIFF 파일을 찾아

    file stem -> 실제 TIFF 경로

    형태로 index를 생성한다.
    """

    print(f"\n[TIFF INDEX] {split_root}")

    tiff_index = {}

    for path in split_root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".tif",
            ".tiff",
        }:
            continue

        stem = path.stem

        if stem in tiff_index:
            raise ValueError(
                f"중복 TIFF stem 발견: {stem}"
            )

        tiff_index[stem] = path

    print(
        f"  TIFF files: "
        f"{len(tiff_index):,}"
    )

    return tiff_index


# ============================================================
# Manifest
# ============================================================

def load_manifest(
    manifest_path: Path,
) -> list[dict]:
    """
    EDA에서 생성한 night_manifest.csv 로드
    """

    rows = []

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            row["image_id"] = int(
                row["image_id"]
            )

            row["person_count"] = int(
                row["person_count"]
            )

            row["has_person"] = (
                str(row["has_person"])
                .lower()
                == "true"
            )

            rows.append(row)

    return rows


# ============================================================
# COCO Load
# ============================================================

def load_coco_person_data(
    coco_path: Path,
) -> tuple[
    dict[int, dict],
    dict[int, list],
]:
    """
    COCO JSON에서

    image 정보
    Person annotation

    만 추출한다.
    """

    print(
        f"[COCO LOAD] {coco_path}"
    )

    with coco_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        coco = json.load(file)

    categories = coco.get(
        "categories",
        [],
    )

    person_category_ids = {
        category["id"]
        for category in categories
        if (
            category.get(
                "name",
                "",
            ).lower()
            == PERSON_CLASS_NAME
        )
    }

    if not person_category_ids:
        raise ValueError(
            "COCO annotation에서 "
            "person 클래스를 찾지 못했습니다."
        )

    image_map = {
        image["id"]: image
        for image in coco.get(
            "images",
            [],
        )
    }

    person_annotations = (
        defaultdict(list)
    )

    for annotation in coco.get(
        "annotations",
        [],
    ):

        if (
            annotation.get(
                "category_id"
            )
            not in person_category_ids
        ):
            continue

        image_id = annotation.get(
            "image_id"
        )

        person_annotations[
            image_id
        ].append(
            annotation
        )

    return (
        image_map,
        person_annotations,
    )


# ============================================================
# COCO -> YOLO
# ============================================================

def coco_bbox_to_yolo(
    bbox: list,
    image_width: float,
    image_height: float,
) -> tuple[
    float,
    float,
    float,
    float,
] | None:
    """
    COCO bbox

    [x_min, y_min, width, height]

    ->

    YOLO bbox

    [x_center, y_center, width, height]

    모든 값은 0~1 normalized.
    """

    if len(bbox) != 4:
        return None

    x, y, width, height = map(
        float,
        bbox,
    )

    # 이미지 범위 안으로 clipping
    x1 = max(
        0.0,
        x,
    )

    y1 = max(
        0.0,
        y,
    )

    x2 = min(
        image_width,
        x + width,
    )

    y2 = min(
        image_height,
        y + height,
    )

    clipped_width = (
        x2 - x1
    )

    clipped_height = (
        y2 - y1
    )

    if (
        clipped_width <= 0
        or clipped_height <= 0
    ):
        return None

    x_center = (
        x1
        + clipped_width / 2
    ) / image_width

    y_center = (
        y1
        + clipped_height / 2
    ) / image_height

    normalized_width = (
        clipped_width
        / image_width
    )

    normalized_height = (
        clipped_height
        / image_height
    )

    return (
        x_center,
        y_center,
        normalized_width,
        normalized_height,
    )


# ============================================================
# Label Validation
# ============================================================

def validate_yolo_label(
    label_path: Path,
) -> tuple[
    int,
    list[str],
]:
    """
    YOLO label 한 파일 검사

    정상:
    0 x_center y_center width height

    Negative sample이면 빈 txt 파일
    """

    errors = []

    text = label_path.read_text(
        encoding="utf-8",
    ).strip()

    # Negative image
    if not text:
        return 0, errors

    lines = text.splitlines()

    valid_boxes = 0

    for line_number, line in enumerate(
        lines,
        start=1,
    ):

        parts = line.split()

        if len(parts) != 5:

            errors.append(
                f"{label_path.name}:"
                f"{line_number} "
                f"field count != 5"
            )

            continue

        try:

            class_id = int(
                parts[0]
            )

            x_center = float(
                parts[1]
            )

            y_center = float(
                parts[2]
            )

            width = float(
                parts[3]
            )

            height = float(
                parts[4]
            )

        except ValueError:

            errors.append(
                f"{label_path.name}:"
                f"{line_number} "
                f"invalid number"
            )

            continue

        if class_id != 0:

            errors.append(
                f"{label_path.name}:"
                f"{line_number} "
                f"class_id != 0"
            )

        coordinates = [
            x_center,
            y_center,
            width,
            height,
        ]

        if not all(
            0.0 <= value <= 1.0
            for value in coordinates
        ):

            errors.append(
                f"{label_path.name}:"
                f"{line_number} "
                f"coordinate outside 0~1"
            )

        if (
            width <= 0
            or height <= 0
        ):

            errors.append(
                f"{label_path.name}:"
                f"{line_number} "
                f"invalid bbox size"
            )

        valid_boxes += 1

    return (
        valid_boxes,
        errors,
    )


# ============================================================
# TIFF Validation
# ============================================================

def validate_tiff_sample(
    image_path: Path,
) -> dict:
    """
    실제 TIFF가 정상적으로 열리는지 확인.
    """

    image = np.array(
        Image.open(
            image_path
        )
    )

    return {
        "file_name": (
            image_path.name
        ),
        "shape": list(
            image.shape
        ),
        "dtype": str(
            image.dtype
        ),
        "min": int(
            image.min()
        ),
        "max": int(
            image.max()
        ),
    }


# ============================================================
# Build One Split
# ============================================================

def build_split(
    split: str,
    manifest_rows: list[dict],
    split_root: Path,
    coco_path: Path,
    output_root: Path,
) -> dict:
    """
    하나의 split에 대해

    TIFF 복사
    +
    YOLO label 생성

    수행.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        f"BUILD SPLIT: {split.upper()}"
    )

    print(
        "=" * 70
    )

    split_manifest = [
        row
        for row in manifest_rows
        if row["split"] == split
    ]

    print(
        f"Night images: "
        f"{len(split_manifest):,}"
    )

    # --------------------------------------------------------
    # Source Index
    # --------------------------------------------------------

    tiff_index = build_tiff_index(
        split_root
    )

    (
        image_map,
        person_annotations,
    ) = load_coco_person_data(
        coco_path
    )

    # --------------------------------------------------------
    # 모든 TIFF가 실제 존재하는지
    # 먼저 확인
    # --------------------------------------------------------

    missing_tiff = []

    for row in split_manifest:

        stem = Path(
            row["file_name"]
        ).stem

        if stem not in tiff_index:

            missing_tiff.append(
                stem
            )

    if missing_tiff:

        print(
            "\n[ERROR] "
            "TIFF mapping 실패"
        )

        print(
            f"Missing TIFF: "
            f"{len(missing_tiff):,}"
        )

        for stem in missing_tiff[:20]:
            print(
                f" - {stem}"
            )

        raise RuntimeError(
            "일부 night image에 "
            "대응하는 TIFF가 없습니다."
        )

    # --------------------------------------------------------
    # Output Folder
    # --------------------------------------------------------

    image_output_dir = (
        output_root
        / "source_tiff"
        / split
    )

    label_output_dir = (
        output_root
        / "labels"
        / split
    )

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    positive_images = 0
    negative_images = 0
    total_person_boxes = 0
    skipped_boxes = 0
    copied_size_bytes = 0

    for index, row in enumerate(
        split_manifest,
        start=1,
    ):

        image_id = row[
            "image_id"
        ]

        image_info = (
            image_map[
                image_id
            ]
        )

        width = float(
            image_info["width"]
        )

        height = float(
            image_info["height"]
        )

        stem = Path(
            row["file_name"]
        ).stem

        source_tiff = (
            tiff_index[
                stem
            ]
        )

        # ----------------------------------------------------
        # TIFF Copy
        # ----------------------------------------------------

        destination_tiff = (
            image_output_dir
            / source_tiff.name
        )

        shutil.copy2(
            source_tiff,
            destination_tiff,
        )

        copied_size_bytes += (
            destination_tiff
            .stat()
            .st_size
        )

        # ----------------------------------------------------
        # YOLO Label
        # ----------------------------------------------------

        annotations = (
            person_annotations.get(
                image_id,
                [],
            )
        )

        yolo_lines = []

        for annotation in annotations:

            converted = (
                coco_bbox_to_yolo(
                    annotation[
                        "bbox"
                    ],
                    width,
                    height,
                )
            )

            if converted is None:

                skipped_boxes += 1
                continue

            (
                x_center,
                y_center,
                bbox_width,
                bbox_height,
            ) = converted

            # class 0 = person
            line = (
                "0 "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{bbox_width:.6f} "
                f"{bbox_height:.6f}"
            )

            yolo_lines.append(
                line
            )

        label_path = (
            label_output_dir
            / f"{stem}.txt"
        )

        # 사람이 없으면 빈 txt
        label_path.write_text(
            "\n".join(
                yolo_lines
            ),
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        if yolo_lines:

            positive_images += 1

        else:

            negative_images += 1

        total_person_boxes += len(
            yolo_lines
        )

        if (
            index % 250 == 0
            or index
            == len(split_manifest)
        ):

            print(
                f"  {index:,} / "
                f"{len(split_manifest):,}"
            )

    return {
        "split": split,
        "images": len(
            split_manifest
        ),
        "positive_images": (
            positive_images
        ),
        "negative_images": (
            negative_images
        ),
        "person_boxes": (
            total_person_boxes
        ),
        "skipped_boxes": (
            skipped_boxes
        ),
        "size_mb": (
            copied_size_bytes
            / 1024
            / 1024
        ),
    }


# ============================================================
# Validate One Split
# ============================================================

def validate_processed_split(
    dataset_root: Path,
    split: str,
    expected_manifest: list[dict],
) -> dict:
    """
    생성된 processed dataset의
    이미지/라벨 무결성을 검사한다.
    """

    image_dir = (
        dataset_root
        / "source_tiff"
        / split
    )

    label_dir = (
        dataset_root
        / "labels"
        / split
    )

    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in {
                    ".tif",
                    ".tiff",
                }
            )
        ]
    )

    label_paths = sorted(
        label_dir.glob(
            "*.txt"
        )
    )

    image_stems = {
        path.stem
        for path in image_paths
    }

    label_stems = {
        path.stem
        for path in label_paths
    }

    missing_labels = (
        image_stems
        - label_stems
    )

    missing_images = (
        label_stems
        - image_stems
    )

    positive_images = 0
    negative_images = 0
    total_boxes = 0

    label_errors = []

    for label_path in label_paths:

        (
            box_count,
            errors,
        ) = validate_yolo_label(
            label_path
        )

        total_boxes += (
            box_count
        )

        if box_count > 0:

            positive_images += 1

        else:

            negative_images += 1

        label_errors.extend(
            errors
        )

    # --------------------------------------------------------
    # Expected count from manifest
    # --------------------------------------------------------

    split_manifest = [
        row
        for row in expected_manifest
        if row["split"] == split
    ]

    expected_images = len(
        split_manifest
    )

    expected_positive = sum(
        row["has_person"]
        for row in split_manifest
    )

    expected_negative = (
        expected_images
        - expected_positive
    )

    expected_boxes = sum(
        row["person_count"]
        for row in split_manifest
    )

    # --------------------------------------------------------
    # Sample TIFF
    # --------------------------------------------------------

    sample_tiff = None

    if image_paths:

        sample_tiff = (
            validate_tiff_sample(
                image_paths[0]
            )
        )

    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------

    checks = {
        "image_count": (
            len(image_paths)
            == expected_images
        ),

        "label_count": (
            len(label_paths)
            == expected_images
        ),

        "positive_count": (
            positive_images
            == expected_positive
        ),

        "negative_count": (
            negative_images
            == expected_negative
        ),

        "person_box_count": (
            total_boxes
            == expected_boxes
        ),

        "image_label_pair": (
            len(missing_labels) == 0
            and len(missing_images) == 0
        ),

        "valid_yolo_labels": (
            len(label_errors) == 0
        ),
    }

    passed = all(
        checks.values()
    )

    return {
        "split": split,

        "expected": {
            "images": (
                expected_images
            ),
            "positive_images": (
                expected_positive
            ),
            "negative_images": (
                expected_negative
            ),
            "person_boxes": (
                expected_boxes
            ),
        },

        "actual": {
            "images": (
                len(image_paths)
            ),
            "labels": (
                len(label_paths)
            ),
            "positive_images": (
                positive_images
            ),
            "negative_images": (
                negative_images
            ),
            "person_boxes": (
                total_boxes
            ),
        },

        "missing_labels": sorted(
            missing_labels
        ),

        "missing_images": sorted(
            missing_images
        ),

        "label_errors": (
            label_errors
        ),

        "sample_tiff": (
            sample_tiff
        ),

        "checks": checks,

        "passed": passed,
    }


# ============================================================
# Console Validation Result
# ============================================================

def print_validation_result(
    result: dict,
) -> None:

    split = result[
        "split"
    ].upper()

    expected = result[
        "expected"
    ]

    actual = result[
        "actual"
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        f"VALIDATION: {split}"
    )

    print(
        "=" * 70
    )

    print(
        f"Expected Images : "
        f"{expected['images']:,}"
    )

    print(
        f"Actual Images   : "
        f"{actual['images']:,}"
    )

    print(
        f"Labels          : "
        f"{actual['labels']:,}"
    )

    print(
        f"Positive        : "
        f"{actual['positive_images']:,}"
    )

    print(
        f"Negative        : "
        f"{actual['negative_images']:,}"
    )

    print(
        f"Person Boxes    : "
        f"{actual['person_boxes']:,}"
    )

    print(
        "\n[CHECKS]"
    )

    for name, status in (
        result[
            "checks"
        ].items()
    ):

        symbol = (
            "PASS"
            if status
            else "FAIL"
        )

        print(
            f"{symbol:4}  {name}"
        )

    if result[
        "sample_tiff"
    ]:

        info = result[
            "sample_tiff"
        ]

        print(
            "\n[TIFF SAMPLE]"
        )

        print(
            f"File  : "
            f"{info['file_name']}"
        )

        print(
            f"Shape : "
            f"{info['shape']}"
        )

        print(
            f"Dtype : "
            f"{info['dtype']}"
        )

        print(
            f"Range : "
            f"{info['min']} "
            f"~ {info['max']}"
        )


# ============================================================
# Build + Validate
# ============================================================

def build_and_validate_dataset(
    raw_root: Path,
    manifest_path: Path,
    output_root: Path,
) -> bool:
    """
    전체 processed dataset 생성 및 검사

    build_processed.py에서
    이 함수 하나만 호출한다.
    """

    print(
        "=" * 70
    )

    print(
        "ThermalSight "
        "Processed Dataset Builder"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Input Check
    # --------------------------------------------------------

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"night_manifest.csv 없음: "
            f"{manifest_path}"
        )

    split_config = {

        "train": {
            "root": (
                raw_root
                / "images_thermal_train"
            ),

            "coco": (
                raw_root
                / "images_thermal_train"
                / "coco.json"
            ),
        },

        "val": {
            "root": (
                raw_root
                / "images_thermal_val"
            ),

            "coco": (
                raw_root
                / "images_thermal_val"
                / "coco.json"
            ),
        },
    }

    for split, config in (
        split_config.items()
    ):

        if not config[
            "root"
        ].exists():

            raise FileNotFoundError(
                f"{split} 데이터 폴더 없음: "
                f"{config['root']}"
            )

        if not config[
            "coco"
        ].exists():

            raise FileNotFoundError(
                f"{split} coco.json 없음: "
                f"{config['coco']}"
            )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest_rows = (
        load_manifest(
            manifest_path
        )
    )

    print(
        f"\nNight Manifest: "
        f"{len(manifest_rows):,} images"
    )

    # --------------------------------------------------------
    # Output Reset
    # --------------------------------------------------------

    if output_root.exists():

        print(
            "\n기존 processed dataset 삭제"
        )

        shutil.rmtree(
            output_root
        )

    (
        output_root
        / "metadata"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    build_results = []

    for split in [
        "train",
        "val",
    ]:

        result = build_split(

            split=split,

            manifest_rows=(
                manifest_rows
            ),

            split_root=(
                split_config[
                    split
                ][
                    "root"
                ]
            ),

            coco_path=(
                split_config[
                    split
                ][
                    "coco"
                ]
            ),

            output_root=(
                output_root
            ),
        )

        build_results.append(
            result
        )

    # --------------------------------------------------------
    # Manifest Copy
    # --------------------------------------------------------

    shutil.copy2(
        manifest_path,
        output_root
        / "metadata"
        / "night_manifest.csv",
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validation_results = []

    for split in [
        "train",
        "val",
    ]:

        validation = (
            validate_processed_split(

                dataset_root=(
                    output_root
                ),

                split=split,

                expected_manifest=(
                    manifest_rows
                ),
            )
        )

        validation_results.append(
            validation
        )

        print_validation_result(
            validation
        )

    all_passed = all(
        result[
            "passed"
        ]
        for result
        in validation_results
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {
        "dataset_name": (
            "ThermalSight "
            "FLIR Night Thermal"
        ),

        "class_mapping": {
            "0": "person",
        },

        "source": {
            "dataset": (
                "FLIR ADAS v2"
            ),

            "subset_condition": (
                "hours == night"
            ),

            "image_type": (
                "high-bit thermal TIFF"
            ),
        },

        "build": (
            build_results
        ),

        "validation": (
            validation_results
        ),

        "all_checks_passed": (
            all_passed
        ),
    }

    report_path = (
        output_root
        / "metadata"
        / "build_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Final Console
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    if all_passed:

        print(
            "ALL CHECKS PASSED"
        )

        print(
            "Processed dataset is ready."
        )

    else:

        print(
            "VALIDATION FAILED"
        )

        print(
            "학습 전에 오류를 "
            "확인해야 합니다."
        )

    print(
        f"\nDataset:"
    )

    print(
        output_root
    )

    print(
        f"\nReport:"
    )

    print(
        report_path
    )

    print(
        "=" * 70
    )

    return all_passed