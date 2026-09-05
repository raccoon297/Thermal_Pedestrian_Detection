# 데이터 카드 (Data Card)

## 1. 기본 정보
- **작성자:** [이름 입력]
- **작성일:** 2026-08-08
- **프로젝트 가제:** ThermalSight

## 2. 데이터 출처
- **데이터셋 이름:** Teledyne FLIR ADAS Thermal Dataset v2
- **출처:** Kaggle mirror  
  https://www.kaggle.com/datasets/samdazel/teledyne-flir-adas-thermal-dataset-v2
- **다운로드 날짜:** 2026-08-08
- **사용 범위:** Thermal Train / Validation 중 `hours == "night"` 데이터
- **라이선스/이용 조건:** 최종 공개 전 원 데이터셋 및 Kaggle mirror 이용 조건 재확인 필요

## 3. 데이터 구조
- **원본 Thermal 이미지**
  - Train: 10,742장
  - Validation: 1,144장

- **정제 후 사용 데이터**
  - Train: 2,110장
  - Validation: 112장
  - 총 이미지: 2,222장
  - Person Bounding Box: 총 14,540개
  - Negative 이미지: 총 279장

- **이미지 형식:** 640×512, grayscale Thermal TIFF, `uint16`
- **타깃:** `person` Bounding Box
- **최종 라벨 형식:** YOLO
- **클래스:** `0 = person`

## 4. 품질 진단 결과
- 전체 Thermal 데이터 중 프로젝트 범위와 맞는 명확한 야간 데이터만 선별할 필요가 있었음.
- `hours == "unknown"`인 이미지가 많아 이번 프로젝트에서는 제외함.
- 야간 Person Bounding Box 높이 중앙값이 Train 24px, Validation 23px로 작은 보행자가 많음.
- 야간 장면은 `city_street` 비중이 높아 데이터 분포에 편향이 있음.
- Validation 야간 데이터가 112장으로 비교적 적음.

## 5. 정제 로그

| 무엇을 | 왜 | 어떻게 | 영향 |
|---|---|---|---|
| RGB 데이터 제외 | 열영상 처리 프로젝트이기 때문 | Thermal 데이터만 사용 | 데이터 범위 축소 |
| 야간 데이터 추출 | 프로젝트 대상이 야간 보행자 탐지이기 때문 | `hours == "night"` 필터 | Train 2,110 / Val 112 |
| Person 클래스만 사용 | 단일 클래스 객체탐지 문제 | COCO에서 `person`만 추출 | 1 class |
| Negative 이미지 유지 | False Positive 학습에 필요 | Person 없는 야간 이미지도 유지 | 총 279장 |
| TIFF만 별도 저장 | 영상 전처리 실험의 공통 원본 확보 | High-bit Thermal TIFF 복사 | 총 2,222장 |
| COCO → YOLO 변환 | 경량 YOLO 모델 학습 준비 | BBox 좌표 정규화 | Person BBox 14,540개 |
| Train/Val split 유지 | 연속 프레임에 의한 데이터 누수 방지 | FLIR 공식 split 유지 | Train 2,110 / Val 112 |

## 6. 특이사항 및 한계
- 작은 보행자가 많아 경량 객체탐지 모델에서 탐지가 어려울 수 있음.
- `city_street` 중심 데이터이므로 모든 야간 환경으로 결과를 일반화하기 어려움.
- Validation 데이터가 적어 전처리별 성능 차이가 작을 경우 결과 변동성에 주의해야 함.
- TIFF는 `uint16` high-bit-depth thermal image로 확인했으며, 센서의 완전한 Raw ADC 데이터라고 단정하지 않음.
- 전처리 기법의 실제 성능 개선 여부는 3회차 모델링에서 검증함.

## 7. 정제 결과 확인
Processed dataset 생성 후 다음 항목을 모두 확인함.

- Image count: PASS
- Label count: PASS
- Positive / Negative count: PASS
- Person Bounding Box count: PASS
- Image / Label pairing: PASS
- YOLO label range: PASS

**최종 판정:** `data/processed/thermal_night/` 데이터셋을 ThermalSight 모델링 단계에 사용한다.
