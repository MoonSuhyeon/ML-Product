"""요청·응답 모델.

예측 응답에 ``as_of`` 를 반드시 싣는다. 이 저장소의 주장이 "예측 시점 기준으로
피처를 재구성했다" 인데, 그 시점이 응답에 없으면 확인할 방법이 없다.
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class ForecastRow(BaseModel):
    property_id: str
    region: str
    property_type: str
    stay_date: date
    as_of: date
    predicted: float
    actual: float | None = None


class ForecastResponse(BaseModel):
    model: str
    horizon_days: int
    as_of: date
    window_start: date
    window_end: date
    count: int
    rows: list[ForecastRow]


class SegmentRow(BaseModel):
    # 축 이름이 아니라 `key` 다. 예전에는 `region` 이었는데, 그러면 "평균 뒤에
    # 가려지는 것을 드러낸다"는 주장이 축 하나에 묶인다. 지역만 볼 수 있는 도구는
    # 숙소 유형별로 무너지는 모델을 못 잡는다.
    key: str
    wape: float
    mae: float
    rmse: float
    zero_ratio: float
    n: int


class SegmentResponse(BaseModel):
    model: str
    # 어느 축으로 쪼갠 결과인지. 응답만 보고도 알 수 있어야 한다.
    dimension: str
    note: str
    rows: list[SegmentRow]


class ModelRow(BaseModel):
    model: str
    wape_mean: float
    wape_std: float
    wape_worst: float
    mae_mean: float
    folds: int
    vs_baseline_pct: float | None = None


class MetricsResponse(BaseModel):
    baseline: str
    serving: str
    measured_by: str
    rows: list[ModelRow]


class LowDemandRow(BaseModel):
    property_id: str
    region: str
    stay_date: date
    predicted: float


class LowDemandResponse(BaseModel):
    threshold: float = Field(description="이 값 미만을 낮은 수요로 본다")
    count: int
    rows: list[LowDemandRow]
