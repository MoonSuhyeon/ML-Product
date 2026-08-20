"""예측 조회.

``/forecast/low-demand`` 가 콘솔에서 실제로 쓰이는 자리다 — 수요가 낮게 나온
숙소·날짜를 집어내면, 그게 콘텐츠 생성(RAG-Marketing)의 대상이 된다.
루프가 이어지는 지점이라 엔드포인트로 만들어 둔다.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from src.evaluation.metrics import evaluate_by
from src.features.asof import DEFAULT_HORIZON

from api.schemas import (
    ForecastResponse, ForecastRow, LowDemandResponse, LowDemandRow,
    SegmentResponse, SegmentRow,
)
from api.state import SERVING_MODEL, runtime

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("", response_model=ForecastResponse)
def forecast(
    region: str | None = Query(None, description="지역으로 거른다"),
    limit: int = Query(200, ge=1, le=2000),
) -> ForecastResponse:
    df = runtime.forecast
    fold = runtime.fold
    if region:
        df = df[df["region"] == region]

    rows = [
        ForecastRow(
            property_id=r.property_id,
            region=r.region,
            property_type=r.property_type,
            stay_date=r.stay_date.date(),
            as_of=r.as_of.date(),
            predicted=round(float(r.pred), 3),
            actual=float(r.demand),
        )
        for r in df.head(limit).itertuples(index=False)
    ]
    return ForecastResponse(
        model=SERVING_MODEL,
        horizon_days=DEFAULT_HORIZON,
        as_of=fold.train_end.date(),
        window_start=fold.valid_start.date(),
        window_end=fold.valid_end.date(),
        count=len(df),
        rows=rows,
    )


@router.get("/low-demand", response_model=LowDemandResponse)
def low_demand(
    threshold: float = Query(1.0, ge=0, description="이 값 미만을 낮은 수요로 본다"),
    region: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> LowDemandResponse:
    """수요가 낮게 예측된 숙소·날짜.

    콘솔과 조정자가 이걸 받아 개입 대상으로 넘긴다.

    **예측값과 그 구간의 오차를 같이 낸다.** 낮은 예측 하나만으로는 행동할 수
    없다는 것을 스키마가 강제한다 — 소비자가 오차를 안 보고 결정하려면 필드를
    일부러 무시해야 하고, 그건 코드에 흔적이 남는다.

    거르지는 않는다. "WAPE 가 높으니 이 행은 빼자" 는 **소비자의 정책**이지
    예측 서비스가 대신 정할 일이 아니다. 여기서 걸러 버리면 조정자는 자기가 무엇을
    못 봤는지조차 모른다.
    """
    df = runtime.forecast
    if region:
        df = df[df["region"] == region]
    low = df[df["pred"] < threshold].sort_values("pred")

    # 지역별 오차를 **예측과 같은 폴드에서** 잰다. 다른 구간에서 잰 값을 붙이면
    # "이 예측이 얼마나 미더운가" 가 아니라 "예전에 얼마나 미더웠나" 가 된다.
    seg = evaluate_by(runtime.forecast, "region")
    wape_by = {str(r.region): float(r.wape) for r in seg.itertuples(index=False)}
    n_by = {str(r.region): int(r.n) for r in seg.itertuples(index=False)}

    return LowDemandResponse(
        threshold=threshold,
        count=len(low),
        measured_on=runtime.fold.valid_start.date(),
        note="region_wape 가 null 이면 그 지역을 잴 표본이 없었다는 뜻이다 — 0 이 아니다.",
        rows=[
            LowDemandRow(
                property_id=r.property_id,
                region=r.region,
                stay_date=r.stay_date.date(),
                predicted=round(float(r.pred), 3),
                region_wape=(round(wape_by[str(r.region)], 4)
                             if str(r.region) in wape_by else None),
                region_n=n_by.get(str(r.region), 0),
            )
            for r in low.head(limit).itertuples(index=False)
        ],
    )


# 쪼갤 수 있는 축. 패널에 있는 컬럼만 허용한다 — 임의 문자열을 그대로
# `groupby` 에 넘기면 사용자가 KeyError 를 보게 된다.
SEGMENT_DIMENSIONS = ("region", "property_type")


@router.get("/segments", response_model=SegmentResponse)
def segments(
    by: str = Query("region", description=f"쪼갤 축. {' | '.join(SEGMENT_DIMENSIONS)}"),
) -> SegmentResponse:
    """세그먼트별 오차.

    전체 WAPE 만 보면 희박한 구간이 가려진다. 평균 뒤에 무엇이 숨어 있는지를
    같이 내보내는 게 이 프로젝트의 규칙이라 API 에서도 그대로 한다.

    **축이 하나면 그 규칙이 반쪽이다.** 지역만 볼 수 있으면 특정 숙소 유형에서만
    무너지는 모델을 지역 평균이 덮어 가린다. 그래서 축을 파라미터로 연다.
    """
    if by not in SEGMENT_DIMENSIONS:
        raise HTTPException(400, f"쪼갤 수 없는 축입니다: {by}")

    seg = evaluate_by(runtime.forecast, by)
    return SegmentResponse(
        model=SERVING_MODEL,
        dimension=by,
        note="가장 최근 폴드 기준. 전체 평균 뒤에 가려지는 구간 편차를 드러낸다.",
        rows=[
            SegmentRow(
                key=str(getattr(r, by)),
                wape=round(float(r.wape), 4),
                mae=round(float(r.mae), 4),
                rmse=round(float(r.rmse), 4),
                zero_ratio=round(float(r.zero_ratio), 4),
                n=int(r.n),
            )
            for r in seg.itertuples(index=False)
        ],
    )
