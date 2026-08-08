# ML-Product

> Airbnb형 숙박 플랫폼의 **지역·숙소별 예약 수요를 예측하고, 가격·시즌·예약 이력 등 다양한 요인을 반영하여 수요 예측 모델을 고도화하는 ML 프로젝트**

숙박 플랫폼에서 발생하는 예약 데이터를 기반으로 **지역–숙소 단위의 미래 예약 수요를 예측**하고, 예측 결과를 가격·프로모션·재고·운영 의사결정에 활용할 수 있도록 수요 예측 엔진을 설계한 프로젝트입니다.

단순히 하나의 예측 모델을 학습하는 데 그치지 않고, 데이터 탐색 → 시계열 특성 분석 → Feature Engineering → 모델 학습 → 예측 성능 검증 → 지역·숙소 단위 세분화 → 실험 및 개선의 과정을 통해 **서비스에 적용 가능한 수요 예측 구조**를 구현했습니다.

---

## 프로젝트 개요

숙박 예약 수요는 단순한 과거 예약량만으로 결정되지 않습니다.

요일, 시즌, 가격, 예약 리드타임, 지역, 숙소 유형, 과거 예약량 등 다양한 요인이 복합적으로 작용합니다.

따라서 다음과 같은 문제를 정의했습니다.

> **"특정 지역과 숙소에서 향후 예약 수요가 얼마나 발생할 것인가?"**

이를 해결하기 위해 예약 데이터를 시계열 관점에서 분석하고, 지역·숙소별 수요 패턴을 학습하는 예측 모델을 구축했습니다.

```text
예약 데이터
    ↓
데이터 정제
    ↓
시계열 특성 분석
    ↓
Feature Engineering
    │
    ├── 날짜 / 요일
    ├── 시즌
    ├── 가격
    ├── 예약 이력
    ├── 예약 리드타임
    ├── 지역
    └── 숙소 특성
    ↓
수요 예측 모델
    ↓
지역 × 숙소 단위 예측
    ↓
예측 성능 검증
    ↓
수요 예측 결과
    ↓
가격 / 프로모션 / 재고 / 운영 의사결정
```

---

# 주요 업무

### 1. 지역·숙소별 예약 수요 예측 모델 연구·고도화

숙박 플랫폼의 예약 데이터를 기반으로 **지역–숙소 단위의 예약 수요를 예측하는 모델**을 설계했습니다.

전체 플랫폼의 총 예약량만 예측하는 방식에서 벗어나,

```text
전체 플랫폼
    ↓
지역
    ↓
지역별 숙소
    ↓
숙소별 일/주간 예약 수요
```

형태로 예측 단위를 세분화하여 실제 서비스의 운영 의사결정에 활용할 수 있는 구조를 설계했습니다.

---

### 2. 시즌·요일·가격·예약 이력 기반 Feature Engineering

예약 수요에 영향을 줄 수 있는 주요 변수를 분석하고 시계열 Feature를 구성했습니다.

#### 시간 기반 Feature

* 연도
* 월
* 주차
* 요일
* 주말 여부
* 공휴일 여부
* 성수기 / 비수기
* 시즌 구분

#### 예약 이력 Feature

* 전일 예약량
* 최근 7일 예약량
* 최근 14일 예약량
* 최근 28일 예약량
* 이동평균
* 이동 표준편차
* 예약 증가율
* 전주 대비 예약 변화량

#### 숙소 Feature

* 지역
* 숙소 유형
* 객실 수
* 수용 인원
* 편의시설
* 평점
* 리뷰 수

#### 가격 Feature

* 현재 가격
* 평균 가격
* 지역 평균 대비 가격
* 가격 변화율

이를 통해 단순한 `날짜 → 예약량` 예측이 아니라 **예약 수요를 구성하는 다양한 요인을 함께 고려하는 예측 구조**를 구축했습니다.

---

### 3. 시계열 특성 분석 및 예측 정합성 검증

예약 데이터의 시간적 패턴을 분석하여 모델이 학습해야 할 수요 구조를 확인했습니다.

주요 분석 대상:

* 장기 추세
* 요일별 반복 패턴
* 계절성
* 성수기 예약 급증
* 지역별 수요 차이
* 숙소별 수요 편차
* 가격 변화에 따른 예약량 변화
* 이상치 및 급격한 수요 변동

특히 일반적인 Random Train/Test Split을 사용하지 않고 **시간 순서를 유지하는 방식으로 학습·검증 데이터를 분리**하여 미래 정보를 학습 데이터에 포함시키는 데이터 누수(Data Leakage)를 방지했습니다.

```text
Past
──────────────────────→ Future

Train                Validation          Test
███████████████████   ███████             ███████
```

---

### 4. 수요 예측 모델 비교 및 실험

하나의 모델을 고정적으로 사용하는 대신 다양한 접근법을 비교하여 숙박 예약 데이터에 적합한 예측 구조를 검증했습니다.

#### Baseline

* Naive Forecast
* Moving Average
* Seasonal Naive

#### Machine Learning

* Random Forest
* XGBoost

#### Time Series

* ARIMA 계열
* 계절성을 반영한 시계열 모델

모델별 예측 성능을 비교하고 데이터 특성에 따라 어떤 접근법이 효과적인지 실험했습니다.

---

## 예측 목표

기본 예측 단위는 다음과 같이 설정했습니다.

```text
지역 × 숙소 × 날짜
```

예:

```text
서울 × 강남 A숙소 × 2026-08-15
→ 예상 예약 수요: 18건

부산 × 해운대 B숙소 × 2026-08-15
→ 예상 예약 수요: 31건
```

이를 주간 단위로 집계하면:

```text
지역
 ├── 숙소 A
 │    ├── 월 12
 │    ├── 화 15
 │    ├── 수 17
 │    └── ...
 │
 └── 숙소 B
      ├── 월 23
      ├── 화 25
      └── ...
```

형태의 수요 예측 결과를 생성할 수 있도록 구성했습니다.

---

# 모델 평가

수요 예측은 단순히 평균적인 오차가 낮은 것뿐 아니라 **수요가 급증하거나 감소하는 시점을 얼마나 정확하게 예측하는지**가 중요합니다.

따라서 다음 지표를 활용하여 모델을 평가했습니다.

| Metric | 목적              |
| ------ | --------------- |
| MAE    | 평균 절대 오차        |
| RMSE   | 큰 예측 오차에 대한 민감도 |
| MAPE   | 실제 수요 대비 상대적 오차 |
| WAPE   | 전체 예약량 기준 예측 오차 |
| R²     | 설명력 확인          |

특히 숙소별 예약량 규모가 크게 다른 경우를 고려하여 **MAE/RMSE와 함께 상대적인 오차 지표를 병행**하여 모델을 비교했습니다.

---

# 지역·숙소 단위 모델링

전체 데이터를 하나의 모델로 학습하는 방식과 지역·숙소별 특성을 반영하는 방식의 차이를 비교했습니다.

### Global Model

```text
전체 예약 데이터
       ↓
   하나의 모델
       ↓
지역·숙소별 예측
```

### Segment-aware Model

```text
예약 데이터
   ↓
지역 / 숙소 Feature 반영
   ↓
통합 모델
   ↓
지역·숙소별 예측
```

숙소마다 예약량 규모와 계절성이 다르기 때문에 단순히 전체 평균 패턴을 적용하는 것보다 **지역과 숙소의 특성을 Feature로 반영하는 방식**을 중심으로 모델을 고도화했습니다.

---

# 데이터 처리 Pipeline

```text
Raw Reservation Data
        ↓
Data Validation
        ↓
Missing Value 처리
        ↓
Outlier 처리
        ↓
Time Series Aggregation
        ↓
Feature Engineering
        ↓
Train / Validation / Test Split
        ↓
Model Training
        ↓
Prediction
        ↓
Evaluation
        ↓
Model Comparison
        ↓
Forecast Output
```

---

# Feature Engineering Pipeline

```text
예약 이력
    │
    ├── Lag Features
    │      ├── lag_1
    │      ├── lag_7
    │      └── lag_28
    │
    ├── Rolling Features
    │      ├── rolling_mean_7
    │      ├── rolling_mean_14
    │      └── rolling_mean_28
    │
    ├── Calendar Features
    │      ├── weekday
    │      ├── weekend
    │      ├── month
    │      └── season
    │
    ├── Property Features
    │      ├── region
    │      ├── property_type
    │      ├── capacity
    │      └── rating
    │
    └── Price Features
           ├── price
           ├── regional_avg_price
           └── price_change_rate
```

---

# 모델 실험 구조

모델 성능을 한 번 측정하고 종료하는 것이 아니라 **가설 → 실험 → 평가 → 개선** 구조로 반복했습니다.

```text
가설
 ↓
Feature 추가
 ↓
모델 학습
 ↓
Validation
 ↓
성능 비교
 ↓
오류 분석
 ↓
다음 실험
```

예:

```text
Experiment 01
Baseline
↓
Experiment 02
+ 요일 Feature
↓
Experiment 03
+ Lag Feature
↓
Experiment 04
+ Rolling Feature
↓
Experiment 05
+ 가격 Feature
↓
Experiment 06
+ 지역/숙소 Feature
```

이를 통해 어떤 Feature가 실제 예측 성능 개선에 기여하는지 비교했습니다.

---

# 예측 결과 분석

모델의 전체 평균 성능뿐 아니라 **어떤 상황에서 예측이 틀리는지**를 분석했습니다.

주요 Error Analysis 대상:

* 성수기 수요 급증
* 비수기 수요 감소
* 특정 지역의 급격한 수요 변화
* 신규 숙소
* 예약량이 적은 숙소
* 가격 급변 구간
* 이벤트·공휴일
* 이상치 발생 구간

이를 통해 단순한 모델 성능 수치가 아니라 **실제 서비스에서 예측 모델이 취약한 상황을 파악하고 다음 개선 방향을 도출**했습니다.

---

# 서비스 적용 구조

예측 결과는 단순한 분석 결과로 끝내지 않고 숙박 플랫폼의 운영 의사결정에 연결될 수 있도록 설계했습니다.

```text
Reservation Data
       ↓
Demand Forecast Engine
       ↓
┌──────────────────────────────┐
│ 지역·숙소별 미래 예약 수요     │
└──────────────────────────────┘
       ↓
┌────────────┬────────────┬────────────┐
│ 가격 전략   │ 프로모션    │ 운영 계획   │
└────────────┴────────────┴────────────┘
```

예를 들어:

### 수요 증가 예상

```text
예상 예약량 ↑
    ↓
성수기 수요 판단
    ↓
가격 / 프로모션 전략 검토
    ↓
객실 운영 계획 조정
```

### 수요 감소 예상

```text
예상 예약량 ↓
    ↓
비수기 수요 판단
    ↓
프로모션 대상 선정
    ↓
예약 전환 촉진
```

---

# 기술 스택

### Language

* Python

### Data Analysis

* pandas
* NumPy
* SciPy

### Machine Learning

* scikit-learn
* XGBoost

### Time Series

* statsmodels

### Visualization

* Matplotlib
* Plotly

### Development

* Jupyter Notebook
* Google Colab

### Data

* PostgreSQL / CSV 기반 예약 데이터
* 지역·숙소·가격·예약 이력 데이터

---

# 프로젝트 구조

```text
ML-Product/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_time_series_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_baseline.ipynb
│   ├── 05_xgboost_model.ipynb
│   └── 06_model_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── forecasting.py
│   ├── evaluation.py
│   └── visualization.py
│
├── models/
│
├── reports/
│   ├── model_comparison.csv
│   └── error_analysis.csv
│
├── requirements.txt
└── README.md
```

---

# 핵심 설계 원칙

### 1. 시간 순서를 보존한 검증

미래 데이터를 이용해 과거를 예측하는 데이터 누수를 방지하기 위해 시간 순서를 유지한 Train/Validation/Test 구조를 사용합니다.

### 2. Baseline 우선

복잡한 모델을 바로 적용하지 않고 Naive Forecast와 Moving Average 등의 Baseline과 비교하여 모델 고도화의 실제 효과를 검증합니다.

### 3. Feature보다 검증

Feature를 무조건 추가하지 않고 실험을 통해 실제 예측 성능에 기여하는 변수를 확인합니다.

### 4. 전체 평균보다 세그먼트

전체 예약량의 평균적인 패턴보다 **지역·숙소별 수요 차이**를 고려하여 서비스에서 활용 가능한 예측 결과를 만드는 것을 목표로 합니다.

### 5. 예측값보다 의사결정

모델의 정확도 자체를 최종 목표로 두지 않고, 예측 결과가 가격·프로모션·재고·운영 등의 실제 의사결정에 활용될 수 있는지를 중심으로 모델을 설계합니다.

---

# 프로젝트를 통해 해결한 문제

| 문제           | 접근                    |
| ------------ | --------------------- |
| 단순 예약량 예측    | 지역·숙소 단위 세분화          |
| 계절성 반영 부족    | 시즌·요일 Feature         |
| 과거 수요 정보 부족  | Lag / Rolling Feature |
| 가격 영향 미반영    | 가격 관련 Feature         |
| 모델 성능 비교 어려움 | Baseline + 모델 비교      |
| 미래 정보 누수     | Time-based Split      |
| 평균 성능만 확인    | Error Analysis        |
| 모델 결과 활용성 부족 | 가격·프로모션·운영 연결         |

---

# 프로젝트 결과

본 프로젝트를 통해 단순한 시계열 예측 모델 구현을 넘어,

**데이터 → Feature → 예측 → 평가 → 오류 분석 → 모델 개선 → 서비스 적용**

으로 이어지는 ML Product 개발 과정을 구현했습니다.

특히 지역·숙소별 예약 수요의 차이를 반영하여 **숙박 플랫폼에서 활용 가능한 수요 예측 엔진**을 설계하고, 모델 성능을 정량적으로 비교하면서 예측 정확도와 서비스 활용성을 함께 개선하는 것을 목표로 했습니다.

---

# 향후 고도화

### Short-term

* 신규 숙소 Cold Start 문제 대응
* 지역별 계절성 모델링
* 공휴일 및 이벤트 데이터 추가
* Hyperparameter Optimization

### Mid-term

* 가격 탄력성 분석
* 수요 급증/감소 Anomaly Detection
* 숙소별 예측 신뢰구간 제공
* Hierarchical Forecasting

### Long-term

```text
예약 데이터
    ↓
수요 예측
    ↓
가격 최적화
    ↓
프로모션 추천
    ↓
예약 전환
    ↓
새로운 행동 데이터
    ↓
모델 재학습
```

예약 수요 예측을 독립적인 ML 모델로 끝내지 않고, 향후 **가격 최적화·프로모션 추천·전환 최적화 시스템과 연결되는 ML Product 구조**로 확장하는 것을 목표로 합니다.
