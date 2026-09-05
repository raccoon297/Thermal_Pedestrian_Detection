# ThermalSight EDA Summary

## Dataset Overview

| Item | Train | Validation |
|---|---:|---:|
| Total Images | 10,742 | 1,144 |
| Night Images | 2,110 | 112 |
| Night + Person Images | 1,851 | 92 |
| Night Negative Images | 259 | 20 |
| Night Person Bounding Boxes | 14,032 | 508 |

---

## Key Findings

### 1. ThermalSight Dataset Scope

ThermalSight는 `hours == night`로 명확하게 분류된 thermal 이미지를
1차 processed dataset으로 사용한다.

Train과 Validation의 공식 split은 그대로 유지한다.

---

### 2. Negative Samples

Person이 존재하지 않는 야간 이미지도 제거하지 않는다.

- Train Negative: 259
- Validation Negative: 20

해당 이미지는 객체탐지 모델의 false positive를 줄이는
negative sample로 사용한다.

---

### 3. Small Pedestrian Problem

야간 Person Bounding Box 높이 중앙값:

- Train: 24.0px
- Validation: 23.0px

하위 25% Bounding Box 높이:

- Train: 15.0px 이하
- Validation: 15.0px 이하

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
