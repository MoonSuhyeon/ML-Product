# ML-Product

> Airbnb형 숙박 플랫폼의 **지역·숙소별 예약 수요를 예측하고, 예측 결과를 가격·프로모션·재고 의사결정에 연결하는 ML 프로덕트**

숙박 플랫폼의 예약 데이터를 기반으로 지역–숙소 단위의 미래 예약 수요를 예측합니다.
모델 학습에서 끝내지 않고 **FastAPI 예측 서빙 · 성능 모니터링 · 재학습 트리거**까지 포함해, 서비스가 실제로 호출할 수 있는 형태로 만드는 것을 목표로 합니다.

---

## 문서 성격

이 문서는 **구현 명세(Spec)** 입니다. 완성된 결과물이 아니라 만들어 가는 목표 상태를 기술합니다.
각 항목의 진행 상태는 다음 표기로 구분합니다.

| 표기 | 의미 |
|------|------|
| ✅ | 구현 완료 |
| 🔨 | 진행 중 |
| 🆕 | 예정 |

---

## 1. 목적

```text
예약 데이터
    ↓
수요 예측 모델
    ↓
지역 × 숙소 × 날짜 단위 예측
    ↓
가격 / 프로모션 / 재고 / 운영 의사결정
```

"모델 정확도"가 아니라 **의사결정에 쓸 수 있는 예측**을 만드는 것이 목표입니다.
따라서 다음 세 가지를 함께 갖추는 것을 완료 조건으로 둡니다.

1. 예측값뿐 아니라 **예측 구간**(불확실성)을 제공한다 — 가격 결정에는 점추정만으로 부족
2. **FastAPI로 서빙**되어 다른 서비스가 호출할 수 있다
3. 배포 후 **오차가 악화되면 감지**되고 재학습이 트리거된다

---

## 2. 문제 정의

### 2-1. 예측 대상

```text
지역 × 숙소 × 날짜  →  예상 예약 수요
```

숙박 도메인은 **날짜가 두 종류**라는 점이 예측 설계의 출발점입니다.

| 기준 | 의미 | 용도 |
|------|------|------|
| **예약 생성일** (booking date) | 예약이 발생한 날 | 매출·마케팅 성과 |
| **숙박일** (stay date) | 실제로 묵는 날 | 재고·가격·운영 |

같은 데이터로 만들지만 **목표 변수가 다르고 누수 위험도 다릅니다.** 본 프로젝트는 숙박일 기준 수요를 주 타깃으로 하고, 예약 생성일 기준 수요를 보조 타깃으로 둡니다.

### 2-2. 리드타임 누수 (이 도메인의 핵심 함정) 🆕

예약은 숙박일보다 **먼저** 발생합니다. 따라서 "8월 15일 숙박 수요"를 8월 1일 시점에 예측하려면, 8월 1일까지 들어온 예약만 써야 합니다.

```text
             예측 시점(as-of)          숙박일
                  │                      │
  ────────────────┼──────────────────────┼────────→
                  │                      │
   사용 가능       │   ← 이 구간의 예약은 아직 발생하지 않음 →
   (관측된 예약)   │      학습에 쓰면 누수
```

단순히 숙박일 기준으로 집계하면 **미래에 들어올 예약까지 피처에 섞여 들어갑니다.**
→ 모든 피처를 **as-of 시점 기준으로 재구성**해야 합니다. (Phase 2)

---

## 3. 아키텍처

```text
        Reservation DB (PostgreSQL)
                  │
                  ↓
        [ Python ] ETL / Aggregation
                  │
                  ↓
        as-of Feature Store  (숙박일 × 예측시점)
                  │
                  ↓
        [ Python ] Training Pipeline
          Baseline → RF → XGBoost → Quantile
                  │
                  ↓
             Model Registry
                  │
                  ↓
        [ FastAPI ] Forecast API
          GET  /forecast
          POST /backfill
          GET  /metrics
                  │
        ┌─────────┼──────────┐
        ↓         ↓          ↓
     가격 전략   프로모션    재고 운영
                  │
                  ↓
        [ Python ] Drift Monitor → 재학습 트리거
```

---

## 4. 기술 범위

### 4-1. 검증 설계 — 시간 순서 보존 🔨

일반 Random Split을 쓰지 않고 **walk-forward 검증**을 사용합니다.
또한 예측 시점과 숙박일 사이에 **gap**을 두어 리드타임 누수를 차단합니다.

```text
Past ──────────────────────────────────────→ Future

Train              gap        Validation
████████████████   ░░░░░      ██████
                   ↑
             예측 리드타임만큼 비움
```

| 항목 | 선택 | 이유 |
|------|------|------|
| 분할 방식 | `TimeSeriesSplit` (expanding window) | 시점 역전 방지 |
| gap | 예측 horizon과 동일 | 리드타임 누수 차단 |
| fold 수 | 5 | 계절 구간별 성능 편차 확인 |

> 일반적인 누수 방지 원칙("`fit`은 학습 데이터에만, test에는 `transform`만")은 시계열에서 **시간 축으로 확장**됩니다. 스케일러뿐 아니라 집계·파생변수 전부가 as-of 시점을 넘어서면 안 됩니다.

### 4-2. Feature Engineering 🔨

| 그룹 | 피처 | 비고 |
|------|------|------|
| **Lag** | `lag_1`, `lag_7`, `lag_14`, `lag_28` | as-of 시점 기준 |
| **Rolling** | `rolling_mean_7/14/28`, `rolling_std_7` | 누수 방지 위해 shift 후 계산 |
| **Calendar** | 요일, 주말, 월, 공휴일, 성수기 | 공휴일 테이블 별도 필요 🆕 |
| **Property** | 지역, 숙소 유형, 수용 인원, 평점, 리뷰 수 | |
| **Price** | 현재가, 지역 평균 대비 비율, 가격 변화율 | |
| **Lead time** | 평균 예약 리드타임, 리드타임 분포 | 숙박 도메인 특화 🆕 |

> 절대값보다 **상대값**이 신호가 강합니다. "1박 8만원"보다 "지역 평균 대비 0.7배"가 수요를 설명합니다. 비율 파생변수를 우선 설계합니다.

### 4-3. 평가 지표 🔨

| Metric | 채택 여부 | 이유 |
|--------|-----------|------|
| **WAPE** | ✅ 주력 | 전체 예약량 기준. 숙소별 규모 차이를 흡수 |
| **MAE** | ✅ 보조 | 절대 오차 감각 |
| **RMSE** | ✅ 보조 | 성수기 큰 오차에 민감 |
| **MAPE** | ⚠️ 제한적 | **수요 0인 날이 많아 발산** — 0 제외 구간에서만 참고 |
| R² | ❌ | 시계열에서 해석력 낮음 |

> MAPE를 주지표로 쓰지 않는 이유를 명시하는 것이 이 프로젝트의 판단 포인트입니다. 숙소 단위로 내려가면 예약 0건인 날이 흔합니다.

### 4-4. 모델 비교 🔨

| 단계 | 모델 | 목적 |
|------|------|------|
| Baseline | Seasonal Naive (전주 동요일) | **이걸 못 이기면 의미 없음** |
| Baseline | Moving Average | |
| ML | Random Forest | 비선형·상호작용 포착 |
| ML | XGBoost | 주력 |
| 시계열 | statsmodels (SARIMA) | 단일 숙소 계절성 검증용 |

> 모델 선택은 **교차검증 → 지표 표 → 선택 근거 기록** 순서를 고정합니다. 어떤 피처와 모델이 실제 성능 개선에 기여했는지를 실험 단위로 남기고, 기여가 확인되지 않은 피처는 채택하지 않습니다.

### 4-5. 예측 구간 🆕

가격 결정에는 "18건 예상"보다 "12~25건 (80% 구간)"이 유용합니다.

```text
XGBoost Quantile Regression
  ├── q=0.1  →  하한
  ├── q=0.5  →  중앙 예측
  └── q=0.9  →  상한
```

### 4-6. 계층 예측 (Hierarchical) 🆕

```text
지역
 └── 숙소
      └── 날짜
```

- 2계층(지역 → 숙소)으로 한정
- **bottom-up reconciliation**만 구현 (숙소 예측 합계 = 지역 예측)
- 검증: 계층 간 합계 정합성 오차 < 1%

### 4-7. Cold Start 🆕

신규 숙소는 과거 예약 이력이 없어 lag 피처가 비어 있습니다.

| 이력 | 전략 |
|------|------|
| 0일 | 지역 × 숙소 유형 × 수용 인원 평균 |
| 1~28일 | 지역 평균과 자체 이력의 가중 혼합 (이력 길이에 비례) |
| 28일+ | 일반 모델 |

### 4-8. 서빙 & 모니터링 🆕

| 엔드포인트 | 설명 |
|-----------|------|
| `GET /forecast?region=&property_id=&start=&end=` | 예측 조회 (배치 결과 캐시) |
| `POST /backfill` | 특정 기간 재예측 |
| `GET /metrics` | 최근 예측 오차 추이 |
| `GET /health` | 헬스체크 |

**Drift 감지**: 최근 7일 WAPE가 학습 시점 대비 임계치 초과 → 재학습 트리거

---

## 5. 구현 로드맵

| Phase | 산출물 | 착수 조건 |
|-------|--------|-----------|
| **0** | 숙박 예약 더미 데이터 생성기 (지역·숙소·계절성·리드타임 반영) | — |
| **1** | EDA 노트북 — 계절성·요일·지역별 수요 분포 확인 | Phase 0 |
| **2** | as-of Feature Store 구축 (`features.py`) | Phase 1 |
| **3** | Baseline(Seasonal Naive) + walk-forward 검증 프레임 | Phase 2 |
| **4** | RF / XGBoost 학습 · 실험 비교표 | Phase 3 |
| **5** | Quantile 예측 구간 + 계층 reconciliation | Phase 4 |
| **6** | FastAPI 서빙 (`/forecast`, `/metrics`) | Phase 4 |
| **7** | Drift 모니터 + 재학습 트리거 | Phase 6 |
| **8** | Cold Start 대응 | Phase 5 |

**Phase 3까지가 최소 완성선**입니다. Baseline을 이기는 모델과 누수 없는 검증 프레임이 있으면 프로젝트로서 성립합니다.

---

## 6. 완료 정의 (DoD)

- [ ] Seasonal Naive Baseline 대비 WAPE 개선폭이 수치로 제시된다
- [ ] walk-forward 검증에서 fold별 성능이 기록된다 (평균만 보고하지 않음)
- [ ] as-of 피처링으로 리드타임 누수가 차단되었음을 테스트로 증명한다
- [ ] `GET /forecast`가 예측값 + 80% 구간을 반환한다
- [ ] 어떤 구간에서 예측이 실패하는지 Error Analysis가 문서화된다
- [ ] 계층 합계 정합성 오차 < 1%

---

## 7. 기술 스택

### 핵심 (Python / FastAPI)

| 영역 | 기술 |
|------|------|
| 언어 | Python 3.11 |
| API | **FastAPI**, Pydantic v2, Uvicorn |
| 데이터 | pandas, NumPy |
| ML | scikit-learn, XGBoost |
| 시계열 | statsmodels |
| DB | PostgreSQL, SQLAlchemy 2.0 (async), Alembic |
| 배치 | APScheduler |
| 테스트 | pytest |
| 시각화 | Matplotlib, Plotly |
| 개발 | Jupyter Notebook |

### 선택

- MLflow — 실험 추적 (Phase 4 이후 도입 검토)
- Streamlit — 내부 예측 확인용 화면

---

## 8. 프로젝트 구조 (목표)

```text
ML-Product/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_baseline.ipynb
│   └── 04_model_comparison.ipynb
├── src/
│   ├── data/
│   │   ├── generator.py        # 더미 예약 데이터 생성
│   │   └── loader.py
│   ├── features/
│   │   ├── asof.py             # as-of 피처링 (핵심)
│   │   ├── calendar.py
│   │   └── price.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── tree.py
│   │   └── quantile.py
│   ├── evaluation/
│   │   ├── metrics.py          # WAPE 등
│   │   ├── walkforward.py
│   │   └── error_analysis.py
│   ├── hierarchy/
│   │   └── reconcile.py
│   └── serving/
│       ├── main.py             # FastAPI 진입점
│       ├── routers/
│       │   ├── forecast.py
│       │   └── metrics.py
│       └── schemas.py
├── tests/
├── reports/
├── requirements.txt
└── README.md
```

---

## 9. 핵심 설계 원칙

1. **시간 순서 보존** — 미래 정보가 학습에 섞이지 않도록 as-of 기준으로 피처를 만든다
2. **Baseline 우선** — Seasonal Naive를 못 이기는 모델은 채택하지 않는다
3. **Feature보다 검증** — 피처를 늘리기 전에 검증 프레임이 옳은지 확인한다
4. **평균보다 세그먼트** — 전체 WAPE가 좋아도 특정 지역·신규 숙소에서 무너지면 실패로 본다
5. **정확도보다 의사결정** — 점추정이 아니라 구간을 제공하고, API로 호출 가능해야 완성이다

---

## 10. 다른 레포와의 연결

| 방향 | 내용 |
|------|------|
| **Data-Growth →** | 이벤트 데이터(검색·조회)를 선행 지표 피처로 수신 |
| **→ Data-Growth** | 수요 예측 결과를 프로모션 실험 대상 선정에 제공 |
| **→ RAG-Marketing** | 비수기 예상 숙소를 마케팅 콘텐츠 생성 대상으로 전달 |

플랫폼 전체 구성은 [프로필 README](https://github.com/MoonSuhyeon)를 참고하세요.
