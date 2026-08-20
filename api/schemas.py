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
    """수요가 낮게 예측된 숙소·날짜 하나.

    **예측값만 실어 보내지 않는다.** 이 응답의 소비자는 사람이 아니라 조정자이고,
    조정자는 이 행을 받아 할인을 승인할지 사람에게 넘길지를 정한다. 그런데
    "0.4개 예약" 이라는 숫자 하나만으로는 그 결정을 할 수 없다 — 그 구간에서
    모델이 평균 40% 씩 틀린다면 0.4 는 사실상 아무 말도 아니기 때문이다.

    그래서 그 구간의 오차를 **같이 싣는다.** 옵션이 아니라 필수 필드다.
    """

    property_id: str
    region: str
    stay_date: date
    predicted: float

    #: 이 지역에서 모델이 얼마나 틀리는가(WAPE). **`None` 은 0 이 아니라 '모른다' 다.**
    #:
    #: 검증 구간에 이 지역 표본이 없으면 잴 수 없다. 그때 0 을 채우면 소비자는
    #: "오차가 없다" 로 읽고 가장 못 믿을 예측을 가장 자신 있게 실행한다.
    #: 모른다는 것은 **낮은 자율성으로 다뤄야 할 사실**이지 결측이 아니다.
    region_wape: float | None = Field(
        description="이 지역의 WAPE. null 이면 잴 표본이 없었다는 뜻 — 0 이 아니다")
    #: WAPE 를 몇 건으로 쟀는가. 3건으로 잰 0.05 와 3,000건으로 잰 0.05 는 다르다.
    region_n: int = Field(description="이 지역의 검증 표본 수")


class LowDemandResponse(BaseModel):
    threshold: float = Field(description="이 값 미만을 낮은 수요로 본다")
    count: int
    #: WAPE 를 잰 구간. 예측과 오차가 **같은 폴드**에서 나왔음을 소비자가 알아야 한다.
    measured_on: date = Field(description="오차를 잰 검증 구간의 시작일")
    note: str = ""
    rows: list[LowDemandRow]
