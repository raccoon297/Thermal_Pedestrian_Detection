# ThermalSight 문제정의서

## 1. 기본 정보

- **작성자:** [이름 입력]
- **작성일:** 2026-08-08
- **프로젝트 가제:** **ThermalSight**
- **데이터 유형:** 비정형 데이터 — 열영상 이미지 / 객체탐지 도전 트랙

---

## 2. 문제 (핵심 질문)

> **FLIR ADAS v2의 야간 열영상에서 영상 전처리 기법을 적용한 입력을 사용했을 때, 동일한 경량 객체탐지 모델의 보행자 탐지 성능을 개선할 수 있는가?**

자동차·방산과 같은 임베디드 시스템에서는 연산 자원과 처리시간의 제약 때문에 무거운 AI 모델을 무작정 사용할 수 없다.

따라서 모델 자체의 크기를 키우는 대신 **열영상의 대비와 객체 특징을 영상 신호 처리로 개선하여, 동일한 경량 객체탐지 모델의 성능을 높일 수 있는지** 실험한다.

프로젝트의 핵심은 **“어떤 모델이 가장 좋은가?”가 아니라 “같은 경량 모델에서 열영상 입력을 어떻게 처리해야 더 잘 탐지할 수 있는가?”**이다.

---

## 3. 입력과 타깃

### 입력 (모델에게 주는 정보)

- FLIR ADAS v2의 야간 Thermal Image
- High-bit-depth thermal TIFF를 기반으로 생성한 전처리별 이미지
- Baseline과 여러 영상처리 pipeline을 동일 조건에서 비교

### 타깃 (모델이 맞혀야 할 것)

- 열영상 속 **보행자(Person)의 위치**
- Bounding Box `(x, y, width, height)`
- 객체탐지 모델이 예측하는 `Person` 클래스 및 confidence

실제 `coco.json`에서 `person`이 `category id = 1`로 존재하며, 각 객체에 Bounding Box annotation이 포함되어 있음을 확인했다.

따라서 이번 프로젝트는 단순 이미지 분류가 아니라 **객체탐지(Object Detection)** 문제다.

---

## 4. 선정 배경 (왜 이 문제인가)

### 개인적 동기

아이쓰리시스템의 영상 알고리즘 개발 직무가 **영상 알고리즘 개발, 열영상 영상처리, AI 개발, 임베디드 코딩 개발**을 주요 업무로 하고 있다는 점에서 프로젝트를 출발했다.

특히 실제 자동차·방산·감시 시스템에서는 높은 성능만을 위해 모델을 계속 대형화하기 어렵고, 제한된 연산 자원에서 동작할 수 있는 경량 AI 모델이 중요하다고 판단했다.

따라서 **모델의 복잡도를 증가시키는 대신 열영상 입력 자체를 효과적으로 처리하여 경량 객체탐지 모델의 성능을 보완할 수 있는지** 확인하고자 한다.

이 프로젝트를 통해 단순히 객체탐지 모델을 사용하는 것에 그치지 않고,

**열영상 → 영상처리 → AI 추론 → 정확도/처리속도 평가**

까지 연결된 영상 AI 파이프라인을 경험하는 것을 목표로 한다.

### 수혜자 한 문장

> **이 시스템은 제한된 연산 자원을 가진 열영상 기반 차량·감시·임베디드 시스템 개발자가, 모델 크기를 증가시키지 않고 영상처리 방법을 통해 보행자 탐지 성능을 개선하는 데 도움이 됩니다.**

---

## 5. 데이터 계획

| 데이터 | 출처 | 확인 내용 | 확인 여부 |
|---|---|---|---|
| FLIR ADAS v2 Thermal Train | Teledyne FLIR ADAS Thermal Dataset v2 / Kaggle mirror | Thermal 이미지 존재 | ✅ |
| High-bit Thermal TIFF | FLIR ADAS v2 | 640×512, `uint16` TIFF 실제 확인 | ✅ |
| Thermal JPG | FLIR ADAS v2 | 640×512, 8-bit JPG 존재 | ✅ |
| 객체탐지 Annotation | `images_thermal_train/coco.json` | COCO Bounding Box 형식 | ✅ |
| Person Target | `coco.json` | `person`, category id=1 존재 | ✅ |
| 야간 정보 | `coco.json` image metadata | `hours: "night"` 존재 | ✅ |
| 장면 정보 | `coco.json` image metadata | `city_street`, `residential` 등 존재 | ✅ |
| Train/Validation | FLIR ADAS v2 | thermal train / thermal val 분리 | ✅ |
| 라이선스·재배포 조건 | 원 데이터셋/배포 페이지 | 최종 포트폴리오 공개 전 확인 | ⏳ |

### 확인된 Train 데이터 규모

- 전체 Train 이미지: **10,742장**
- 전체 Annotation: **175,040개**
- Person Bounding Box: **50,478개**
- `hours = night` 이미지: **2,110장**
- 야간 이미지 중 Person 포함 이미지: **1,851장**
- 야간 Person Bounding Box: **14,032개**
- Person이 없는 야간 이미지: **259장**

야간 이미지 metadata에 `hours: "night"`가 기록되어 있고, `city_street`, `residential`, `parking_lot`, `highway` 등의 scene 정보도 포함되어 있어 야간 환경 subset 구성이 가능하다.

---

## 6. 예상 접근 방법 (가설)

### 핵심 가설

> **열영상의 대비와 보행자 특징을 강화하는 적절한 영상 전처리를 적용하면, 모델의 크기를 증가시키지 않고도 동일한 경량 객체탐지 모델의 보행자 Recall과 mAP를 개선할 수 있을 것이다.**

단, 영상처리가 복잡해질수록 전처리 시간이 증가할 수 있으므로 **탐지 성능이 가장 높은 방법이 반드시 실제 시스템에서 가장 좋은 방법은 아닐 것**이라고 예상한다.

따라서 정확도와 처리속도를 함께 평가한다.

### 기본 실험 설계

모델은 **경량 객체탐지 모델 1종**으로 고정한다.

예:

> YOLO nano급 경량 Detector  
> *(정확한 모델 버전은 3회차 모델링 단계에서 확정)*

실험 조건:

```text
[Experiment 0 — Baseline]

High-bit Thermal
→ 기본 정규화 / 변환
→ Lightweight Detector


[Experiment 1]

High-bit Thermal
→ Contrast Stretching
→ Lightweight Detector


[Experiment 2]

High-bit Thermal
→ CLAHE
→ Lightweight Detector


[Experiment 3]

High-bit Thermal
→ Denoising + CLAHE
→ Lightweight Detector
```

모든 실험에서 다음 조건을 동일하게 유지한다.

- 동일한 모델 구조
- 동일한 Train / Validation split
- 동일한 epoch
- 동일한 image size
- 동일한 optimizer 및 주요 학습 설정
- 가능하면 동일한 random seed

**영상 전처리 방법만 변경하여 성능 차이를 비교한다.**

### 주요 평가 지표

#### 객체탐지 성능

- Precision
- Recall
- mAP50
- 필요 시 mAP50-95

#### 실시간성 / 경량 시스템 관점

- 전처리 시간(ms)
- 모델 inference time(ms)
- 전체 처리시간(ms)
- FPS

단순히 최고 mAP를 선택하지 않고,

> **탐지 성능 향상 대비 추가 연산비용**

을 함께 비교하여 최종 preprocessing pipeline을 선정한다.

---

## 7. 예상 결과물 (4회차 결과 상상하기)

### 프로젝트 제목 후보

> **ThermalSight: 열영상 전처리를 통한 경량 AI 보행자 탐지 성능 개선**

또는

> **Can Better Thermal Input Make Lightweight AI Smarter?**  
> 열영상 전처리에 따른 경량 보행자 객체탐지 성능 분석

### 핵심 시각 자료

**전처리 방법별 탐지 성능과 처리시간 Trade-off 그래프**

최종적으로 **“가장 높은 정확도”가 아니라 “정확도 향상 대비 연산비용이 가장 좋은 방법”**을 선정하는 그림을 핵심 결과로 제시한다.

### 함께 보여줄 결과

- Original thermal image
- Contrast Stretching 결과
- CLAHE 결과
- Denoising + CLAHE 결과
- 각 영상에서 동일 Detector가 찾은 Person Bounding Box 비교
- 전처리별 Precision / Recall / mAP
- 전처리별 latency / FPS

### MVP 구상

```text
사용자가 Thermal Image 업로드
            ↓
전처리 방식 선택

[Baseline]
[Contrast]
[CLAHE]
[Denoise + CLAHE]

            ↓
경량 객체탐지 모델 추론
            ↓
Bounding Box가 그려진 결과 영상
            ↓

탐지 인원 수
Confidence
전처리 시간
Inference Time
전체 FPS
```

---

## 8. 리스크와 대안

### 리스크 1 — 전처리가 성능을 향상시키지 않을 수 있음

열영상 전처리를 적용한다고 반드시 객체탐지 성능이 증가한다는 보장은 없다.

오히려 과도한 contrast enhancement나 sharpening으로 인해 thermal image의 특징이 왜곡되어 성능이 감소할 수도 있다.

**Plan B**

성능이 향상되지 않더라도 실패로 간주하지 않고, **“어떤 영상처리는 왜 경량 Detector에 도움이 되지 않았는가?”**를 분석한다.

예:

- CLAHE → 작은 객체의 contrast 향상
- 과도한 sharpening → noise까지 강조
- Denoising → 작은 보행자 특징까지 제거

---

### 리스크 2 — 작은 보행자 객체

FLIR 야간 이미지에는 화면에서 크기가 작은 보행자가 많이 포함되어 있어 경량 Detector가 탐지에 어려움을 겪을 가능성이 있다.

**Plan B**

- 원본 해상도를 지나치게 축소하지 않는다.
- 너무 작은 Bounding Box를 별도로 분석한다.
- 전체 성능뿐 아니라 **small pedestrian 영역에서 전처리 효과가 달라지는지** 추가 분석한다.

---

### 리스크 3 — 전처리 경우의 수 증가

영상처리 방법을 지나치게 많이 비교하면 4주 미니프로젝트 범위를 넘어갈 수 있다.

**Plan B**

최종 실험은 우선 다음 세 가지 계열로 제한한다.

1. Baseline
2. Contrast / CLAHE 계열
3. Denoising + Contrast 계열

초기 결과가 나온 뒤 필요할 경우에만 추가 실험한다.

---

### 리스크 4 — 경량 모델의 절대 성능이 낮을 가능성

경량 모델 특성상 대형 Detector보다 절대적인 성능은 낮을 수 있다.

하지만 이번 프로젝트의 목적은 대형 모델을 이기는 것이 아니다.

> **동일한 경량 모델을 사용하면서 영상처리를 통해 얼마만큼 성능을 보완할 수 있는가**

를 검증하는 것이 핵심이다.

따라서 대형 모델과의 경쟁보다 **Baseline 대비 개선폭**을 중심으로 결과를 해석한다.

---

## 프로젝트 핵심 한 문장

> **ThermalSight는 모델을 더 크게 만드는 대신, 열영상 입력을 더 잘 처리함으로써 제한된 연산 환경의 경량 AI 보행자 탐지 성능을 개선할 수 있는지 검증하는 프로젝트다.**
