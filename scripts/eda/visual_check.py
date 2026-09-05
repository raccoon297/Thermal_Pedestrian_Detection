import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle


PERSON_CLASS_NAME = "person"


def build_tiff_index(
    split_root: Path,
) -> dict[str, Path]:
    """
    split 폴더 내부의 TIFF 파일을 전부 탐색해서

    파일 stem -> 실제 TIFF 경로

    형태의 index 생성
    """

    index = {}

    for path in split_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".tif",
            ".tiff",
        }:
            continue

        stem = path.stem

        if stem in index:
            raise ValueError(
                f"중복 TIFF stem 발견: {stem}"
            )

        index[stem] = path

    return index


def normalize_tiff_for_display(
    image: np.ndarray,
) -> np.ndarray:
    """
    High-bit thermal TIFF를
    사람이 보기 좋은 8-bit 이미지로 변환.

    단순 min-max 대신 1~99 percentile을 사용해서
    극단값의 영향을 줄인다.

    이 변환은 EDA 시각화 전용이며,
    실제 모델 전처리 방식은 아님.
    """

    image = image.astype(
        np.float32
    )

    low = np.percentile(
        image,
        1,
    )

    high = np.percentile(
        image,
        99,
    )

    if high <= low:
        low = image.min()
        high = image.max()

    if high <= low:
        return np.zeros_like(
            image,
            dtype=np.uint8,
        )

    image = np.clip(
        image,
        low,
        high,
    )

    image = (
        (image - low)
        / (high - low)
        * 255
    )

    return image.astype(
        np.uint8
    )


def create_visual_check(
    coco_path: Path,
    split_root: Path,
    save_path: Path,
    sample_count: int = 6,
    seed: int = 42,
) -> None:
    """
    야간 + Person 이미지 중 일부를 랜덤 선택하여

    실제 High-bit TIFF
    +
    COCO Person Bounding Box

    를 함께 시각화한다.
    """

    print(
        "\n[Visual Check] "
        "실제 thermal TIFF 확인 중..."
    )

    with coco_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        coco = json.load(file)

    images = coco["images"]
    annotations = coco[
        "annotations"
    ]
    categories = coco[
        "categories"
    ]

    # ----------------------------------------------------
    # Person Category
    # ----------------------------------------------------

    person_ids = {
        category["id"]
        for category in categories
        if category["name"].lower()
        == PERSON_CLASS_NAME
    }

    if not person_ids:
        raise ValueError(
            "person 클래스를 찾을 수 없습니다."
        )

    # ----------------------------------------------------
    # Image Metadata
    # ----------------------------------------------------

    image_map = {
        image["id"]: image
        for image in images
    }

    night_image_ids = {
        image["id"]
        for image in images
        if (
            image.get(
                "extra_info"
            )
            or {}
        ).get(
            "hours"
        )
        == "night"
    }

    # ----------------------------------------------------
    # Person Annotation
    # ----------------------------------------------------

    person_boxes = defaultdict(
        list
    )

    for annotation in annotations:

        if (
            annotation[
                "category_id"
            ]
            not in person_ids
        ):
            continue

        image_id = annotation[
            "image_id"
        ]

        if image_id not in night_image_ids:
            continue

        person_boxes[
            image_id
        ].append(
            annotation["bbox"]
        )

    candidate_ids = list(
        person_boxes.keys()
    )

    if len(candidate_ids) < sample_count:
        sample_count = len(
            candidate_ids
        )

    random.seed(seed)

    sampled_ids = random.sample(
        candidate_ids,
        sample_count,
    )

    # ----------------------------------------------------
    # TIFF Index
    # ----------------------------------------------------

    print(
        "[Visual Check] "
        "TIFF index 생성 중..."
    )

    tiff_index = build_tiff_index(
        split_root
    )

    # ----------------------------------------------------
    # Plot
    # ----------------------------------------------------

    cols = 3

    rows = (
        sample_count
        + cols
        - 1
    ) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(15, 5 * rows),
    )

    axes = np.array(
        axes
    ).reshape(-1)

    for ax, image_id in zip(
        axes,
        sampled_ids,
    ):
        image_info = image_map[
            image_id
        ]

        jpg_name = Path(
            image_info[
                "file_name"
            ]
        )

        stem = jpg_name.stem

        tiff_path = tiff_index.get(
            stem
        )

        if tiff_path is None:
            ax.set_title(
                f"TIFF not found\n{stem}"
            )
            ax.axis("off")
            continue

        # High-bit TIFF 로드
        thermal = np.array(
            Image.open(
                tiff_path
            )
        )

        display_image = (
            normalize_tiff_for_display(
                thermal
            )
        )

        ax.imshow(
            display_image,
            cmap="gray",
        )

        # Bounding Box
        for bbox in person_boxes[
            image_id
        ]:
            x, y, width, height = bbox

            rectangle = Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                linewidth=2,
            )

            ax.add_patch(
                rectangle
            )

        extra_info = (
            image_info.get(
                "extra_info"
            )
            or {}
        )

        scene = extra_info.get(
            "scene",
            "unknown",
        )

        ax.set_title(
            f"{scene}\n"
            f"Person: "
            f"{len(person_boxes[image_id])}"
        )

        ax.axis("off")

    # 남는 subplot 제거
    for ax in axes[
        sample_count:
    ]:
        ax.axis("off")

    fig.suptitle(
        "Night Thermal TIFF "
        "with Person Bounding Boxes",
        fontsize=16,
    )

    fig.tight_layout()

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        save_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "[Visual Check] 저장 완료:"
    )

    print(
        save_path
    )