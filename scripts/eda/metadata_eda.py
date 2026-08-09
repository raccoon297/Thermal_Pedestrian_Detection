import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


# ============================================================
# Constant
# ============================================================

PERSON_CLASS_NAME = "person"


# ============================================================
# Utility
# ============================================================

def quantile(
    values: list[float],
    q: float,
) -> float:
    """
    간단한 분위수 계산
    """

    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return float(values[0])

    position = (len(values) - 1) * q

    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    weight = position - lower

    return float(
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    """
    dict list를 CSV로 저장
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dir(
    output_dir: Path,
) -> None:
    """
    이전 EDA 결과 삭제 후
    깨끗한 출력 폴더 생성
    """

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )



# ============================================================
# Load / Analyze COCO
# ============================================================

def analyze_split(
    coco_path: Path,
    split: str,
) -> dict:
    """
    하나의 COCO annotation 파일 분석

    이미지 파일 자체는 읽지 않고
    coco.json metadata만 사용한다.
    """

    print(f"\n[{split.upper()}] COCO JSON loading...")

    with coco_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        coco = json.load(file)

    images = coco.get("images", [])
    annotations = coco.get(
        "annotations",
        [],
    )
    categories = coco.get(
        "categories",
        [],
    )

    print(
        f"[{split.upper()}] JSON load complete"
    )

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    category_map = {
        category["id"]: category["name"]
        for category in categories
    }

    person_category_ids = {
        category_id
        for category_id, name
        in category_map.items()
        if name.lower()
        == PERSON_CLASS_NAME
    }

    if not person_category_ids:
        raise ValueError(
            f"{split}: person 클래스를 "
            "찾을 수 없습니다."
        )

    person_category_id = min(
        person_category_ids
    )

    # --------------------------------------------------------
    # Image metadata
    # --------------------------------------------------------

    image_map = {
        image["id"]: image
        for image in images
    }

    hours_counter = Counter()
    night_scene_counter = Counter()

    night_image_ids = set()

    for image in images:
        extra_info = (
            image.get("extra_info")
            or {}
        )

        hours = extra_info.get(
            "hours",
            "unknown",
        )

        hours_counter[hours] += 1

        if hours == "night":
            image_id = image["id"]

            night_image_ids.add(
                image_id
            )

            scene = extra_info.get(
                "scene",
                "unknown",
            )

            night_scene_counter[
                scene
            ] += 1

    # --------------------------------------------------------
    # Person Annotation
    # --------------------------------------------------------

    person_count_by_image = Counter()

    night_bbox_widths = []
    night_bbox_heights = []
    night_bbox_areas = []

    for annotation in annotations:
        category_id = annotation.get(
            "category_id"
        )

        if category_id not in person_category_ids:
            continue

        image_id = annotation.get(
            "image_id"
        )

        person_count_by_image[
            image_id
        ] += 1

        # 야간 bbox만 분석
        if image_id not in night_image_ids:
            continue

        bbox = annotation.get(
            "bbox",
            [],
        )

        if len(bbox) != 4:
            continue

        _, _, width, height = bbox

        width = float(width)
        height = float(height)

        night_bbox_widths.append(
            width
        )

        night_bbox_heights.append(
            height
        )

        night_bbox_areas.append(
            width * height
        )

    # --------------------------------------------------------
    # Night Manifest
    # --------------------------------------------------------

    night_manifest = []

    for image_id in sorted(
        night_image_ids
    ):
        image = image_map[image_id]

        extra_info = (
            image.get("extra_info")
            or {}
        )

        person_count = (
            person_count_by_image[
                image_id
            ]
        )

        row = {
            "split": split,
            "image_id": image_id,
            "file_name": image.get(
                "file_name",
                "",
            ),
            "width": image.get(
                "width",
                "",
            ),
            "height": image.get(
                "height",
                "",
            ),
            "scene": extra_info.get(
                "scene",
                "unknown",
            ),
            "weather": extra_info.get(
                "weather",
                "unknown",
            ),
            "video_id": extra_info.get(
                "video_id",
                "",
            ),
            "has_person": (
                person_count > 0
            ),
            "person_count": person_count,
        }

        night_manifest.append(row)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    night_positive = sum(
        row["has_person"]
        for row in night_manifest
    )

    night_negative = (
        len(night_manifest)
        - night_positive
    )

    summary = {
        "split": split,
        "total_images": len(images),
        "total_annotations": len(
            annotations
        ),
        "person_category_id": (
            person_category_id
        ),
        "night_images": len(
            night_manifest
        ),
        "night_images_with_person": (
            night_positive
        ),
        "night_images_without_person": (
            night_negative
        ),
        "night_person_boxes": len(
            night_bbox_heights
        ),
    }

    # --------------------------------------------------------
    # BBox Summary
    # --------------------------------------------------------

    bbox_summary = {
        "split": split,
        "bbox_count": len(
            night_bbox_heights
        ),

        "width_q05": quantile(
            night_bbox_widths,
            0.05,
        ),

        "width_q25": quantile(
            night_bbox_widths,
            0.25,
        ),

        "width_median": (
            median(
                night_bbox_widths
            )
            if night_bbox_widths
            else 0
        ),

        "height_q05": quantile(
            night_bbox_heights,
            0.05,
        ),

        "height_q25": quantile(
            night_bbox_heights,
            0.25,
        ),

        "height_median": (
            median(
                night_bbox_heights
            )
            if night_bbox_heights
            else 0
        ),

        "area_median": (
            median(
                night_bbox_areas
            )
            if night_bbox_areas
            else 0
        ),
    }

    return {
        "summary": summary,
        "hours": hours_counter,
        "night_scenes": (
            night_scene_counter
        ),
        "night_manifest": (
            night_manifest
        ),
        "bbox_summary": (
            bbox_summary
        ),
        "night_bbox_heights": (
            night_bbox_heights
        ),
    }


# ============================================================
# Visualization 1
# Hours Distribution
# ============================================================

def plot_hours_distribution(
    results: dict,
    save_path: Path,
) -> None:

    # train / val에 실제 존재하는
    # 모든 hours category 수집
    categories = set()

    for split in ["train", "val"]:
        categories.update(
            results[split][
                "hours"
            ].keys()
        )

    preferred_order = [
        "day",
        "dawn/dusk",
        "night",
        "unknown",
    ]

    ordered_categories = [
        category
        for category
        in preferred_order
        if category in categories
    ]

    ordered_categories += sorted(
        categories
        - set(ordered_categories)
    )

    x = list(
        range(
            len(
                ordered_categories
            )
        )
    )

    width = 0.36

    train_values = [
        results["train"]["hours"].get(
            category,
            0,
        )
        for category
        in ordered_categories
    ]

    val_values = [
        results["val"]["hours"].get(
            category,
            0,
        )
        for category
        in ordered_categories
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        [
            value - width / 2
            for value in x
        ],
        train_values,
        width,
        label="Train",
    )

    ax.bar(
        [
            value + width / 2
            for value in x
        ],
        val_values,
        width,
        label="Val",
    )

    ax.set_title(
        "Thermal Image Distribution "
        "by Time of Day"
    )

    ax.set_xlabel(
        "Time of Day"
    )

    ax.set_ylabel(
        "Image Count"
    )

    ax.set_xticks(
        x,
        ordered_categories,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=160,
    )

    plt.close(fig)


# ============================================================
# Visualization 2
# Night Scene Distribution
# ============================================================

def plot_night_scene_distribution(
    results: dict,
    save_path: Path,
) -> None:

    combined = (
        results["train"][
            "night_scenes"
        ]
        + results["val"][
            "night_scenes"
        ]
    )

    scenes = [
        name
        for name, _
        in combined.most_common()
    ]

    train_values = [
        results["train"][
            "night_scenes"
        ].get(
            scene,
            0,
        )
        for scene in scenes
    ]

    val_values = [
        results["val"][
            "night_scenes"
        ].get(
            scene,
            0,
        )
        for scene in scenes
    ]

    x = list(
        range(
            len(scenes)
        )
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        [
            value - width / 2
            for value in x
        ],
        train_values,
        width,
        label="Train",
    )

    ax.bar(
        [
            value + width / 2
            for value in x
        ],
        val_values,
        width,
        label="Val",
    )

    ax.set_title(
        "Night Thermal Images "
        "by Scene"
    )

    ax.set_xlabel(
        "Scene"
    )

    ax.set_ylabel(
        "Image Count"
    )

    ax.set_xticks(
        x,
        scenes,
        rotation=30,
        ha="right",
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=160,
    )

    plt.close(fig)


# ============================================================
# Visualization 3
# Person Presence
# ============================================================

def plot_person_presence(
    results: dict,
    save_path: Path,
) -> None:

    splits = [
        "train",
        "val",
    ]

    person_present = [
        results[split][
            "summary"
        ][
            "night_images_with_person"
        ]
        for split in splits
    ]

    no_person = [
        results[split][
            "summary"
        ][
            "night_images_without_person"
        ]
        for split in splits
    ]

    x = list(
        range(
            len(splits)
        )
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        [
            value - width / 2
            for value in x
        ],
        person_present,
        width,
        label="Person Present",
    )

    ax.bar(
        [
            value + width / 2
            for value in x
        ],
        no_person,
        width,
        label="No Person",
    )

    ax.set_title(
        "Person Presence in "
        "Night Thermal Images"
    )

    ax.set_xlabel(
        "Dataset Split"
    )

    ax.set_ylabel(
        "Image Count"
    )

    ax.set_xticks(
        x,
        [
            "Train",
            "Val",
        ],
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=160,
    )

    plt.close(fig)


# ============================================================
# Visualization 4
# Person BBox Height Distribution
# ============================================================

def plot_bbox_height_distribution(
    results: dict,
    save_path: Path,
) -> None:

    train_heights = (
        results["train"][
            "night_bbox_heights"
        ]
    )

    val_heights = (
        results["val"][
            "night_bbox_heights"
        ]
    )

    all_heights = (
        train_heights
        + val_heights
    )

    if not all_heights:
        return

    # 극단적으로 큰 box 때문에
    # 그래프가 찌그러지는 것을 방지
    max_height = quantile(
        all_heights,
        0.99,
    )

    max_height = max(
        10,
        int(max_height),
    )

    bins = list(
        range(
            0,
            max_height + 6,
            5,
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.hist(
        train_heights,
        bins=bins,
        alpha=0.65,
        label="Train",
    )

    ax.hist(
        val_heights,
        bins=bins,
        alpha=0.65,
        label="Val",
    )

    ax.set_title(
        "Night Person Bounding Box "
        "Height Distribution"
    )

    ax.set_xlabel(
        "Bounding Box Height (pixels)"
    )

    ax.set_ylabel(
        "Person Box Count"
    )

    ax.set_xlim(
        left=0,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=160,
    )

    plt.close(fig)


# ============================================================
# Human-readable EDA Summary
# ============================================================

def write_summary_markdown(
    results: dict,
    save_path: Path,
) -> None:

    train_summary = (
        results["train"][
            "summary"
        ]
    )

    val_summary = (
        results["val"][
            "summary"
        ]
    )

    train_bbox = (
        results["train"][
            "bbox_summary"
        ]
    )

    val_bbox = (
        results["val"][
            "bbox_summary"
        ]
    )

    text = f"""# ThermalSight EDA Summary

## Dataset Overview

| Item | Train | Validation |
|---|---:|---:|
| Total Images | {train_summary['total_images']:,} | {val_summary['total_images']:,} |
| Night Images | {train_summary['night_images']:,} | {val_summary['night_images']:,} |
| Night + Person Images | {train_summary['night_images_with_person']:,} | {val_summary['night_images_with_person']:,} |
| Night Negative Images | {train_summary['night_images_without_person']:,} | {val_summary['night_images_without_person']:,} |
| Night Person Bounding Boxes | {train_summary['night_person_boxes']:,} | {val_summary['night_person_boxes']:,} |

---

## Key Findings

### 1. ThermalSight Dataset Scope

ThermalSight는 `hours == night`로 명확하게 분류된 thermal 이미지를
1차 processed dataset으로 사용한다.

Train과 Validation의 공식 split은 그대로 유지한다.

---

### 2. Negative Samples

Person이 존재하지 않는 야간 이미지도 제거하지 않는다.

- Train Negative: {train_summary['night_images_without_person']:,}
- Validation Negative: {val_summary['night_images_without_person']:,}

해당 이미지는 객체탐지 모델의 false positive를 줄이는
negative sample로 사용한다.

---

### 3. Small Pedestrian Problem

야간 Person Bounding Box 높이 중앙값:

- Train: {train_bbox['height_median']:.1f}px
- Validation: {val_bbox['height_median']:.1f}px

하위 25% Bounding Box 높이:

- Train: {train_bbox['height_q25']:.1f}px 이하
- Validation: {val_bbox['height_q25']:.1f}px 이하

따라서 FLIR 야간 데이터에는 작은 보행자가 많이 존재하며,
경량 객체탐지 모델의 주요 난점 중 하나로 판단한다.

---

## Processed Dataset Plan

다음 단계에서는 `night_manifest.csv`를 기준으로

1. 야간 thermal TIFF 파일 선택
2. Train / Validation split 유지
3. Person Bounding Box만 추출
4. COCO → YOLO label 변환
5. `data/processed/` 저장

을 수행한다.
"""

    save_path.write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# Main EDA Function
# ============================================================

def run_eda(
    coco_paths: dict[str, Path],
    output_dir: Path,
) -> None:

    # 기존 EDA 결과 정리
    prepare_output_dir(
        output_dir
    )

    plot_dir = output_dir

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    results = {
        split: analyze_split(
            coco_path=path,
            split=split,
        )
        for split, path
        in coco_paths.items()
    }

    # ========================================================
    # CSV 1
    # dataset_summary.csv
    # ========================================================

    summary_rows = [
        results["train"]["summary"],
        results["val"]["summary"],
    ]

    write_csv(
        output_dir
        / "dataset_summary.csv",
        summary_rows,
        list(
            summary_rows[0].keys()
        ),
    )

    # ========================================================
    # CSV 2
    # night_scene_counts.csv
    # ========================================================

    scenes = sorted(
        set(
            results["train"][
                "night_scenes"
            ].keys()
        )
        |
        set(
            results["val"][
                "night_scenes"
            ].keys()
        )
    )

    scene_rows = []

    for scene in scenes:
        scene_rows.append(
            {
                "scene": scene,
                "train_count": (
                    results["train"][
                        "night_scenes"
                    ].get(
                        scene,
                        0,
                    )
                ),
                "val_count": (
                    results["val"][
                        "night_scenes"
                    ].get(
                        scene,
                        0,
                    )
                ),
            }
        )

    write_csv(
        output_dir
        / "night_scene_counts.csv",
        scene_rows,
        [
            "scene",
            "train_count",
            "val_count",
        ],
    )

    # ========================================================
    # CSV 3
    # night_bbox_summary.csv
    # ========================================================

    bbox_rows = [
        results["train"][
            "bbox_summary"
        ],
        results["val"][
            "bbox_summary"
        ],
    ]

    write_csv(
        output_dir
        / "night_bbox_summary.csv",
        bbox_rows,
        list(
            bbox_rows[0].keys()
        ),
    )

    # ========================================================
    # CSV 4
    # night_manifest.csv
    # ========================================================

    manifest_rows = (
        results["train"][
            "night_manifest"
        ]
        +
        results["val"][
            "night_manifest"
        ]
    )

    write_csv(
        output_dir
        / "night_manifest.csv",
        manifest_rows,
        [
            "split",
            "image_id",
            "file_name",
            "width",
            "height",
            "scene",
            "weather",
            "video_id",
            "has_person",
            "person_count",
        ],
    )

    # ========================================================
    # Visualization
    # ========================================================

    plot_night_scene_distribution(
    results,
    plot_dir
    / "02_night_scene_distribution.png",
    )

    plot_person_presence(
    results,
    plot_dir
    / "03_night_person_presence.png",
    )

    plot_bbox_height_distribution(
    results,
    plot_dir
    / "04_night_bbox_height_distribution.png",
    )

    # ========================================================
    # Summary Report
    # ========================================================

    write_summary_markdown(
        results,
        output_dir
        / "EDA_SUMMARY.md",
    )

    # ========================================================
    # Console Summary
    # ========================================================

    print("\n" + "=" * 70)
    print("EDA OUTPUT")
    print("=" * 70)

    print("\n[CSV]")
    print("1. dataset_summary.csv")
    print("2. night_scene_counts.csv")
    print("3. night_bbox_summary.csv")
    print("4. night_manifest.csv")

    print("\n[PLOTS]")
    print("1. 01_hours_distribution.png")
    print("2. 02_night_scene_distribution.png")
    print("3. 03_night_person_presence.png")
    print("4. 04_night_bbox_height_distribution.png")

    print("\n[REPORT]")
    print("EDA_SUMMARY.md")
