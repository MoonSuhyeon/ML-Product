"""as-of 피처링 — 이 프로젝트의 핵심.

예약은 숙박일보다 먼저 발생한다. 따라서 "8월 15일 숙박 수요"를 8월 1일에 예측하려면
**8월 1일까지 들어온 예약만** 써야 한다.

숙박일 기준으로 그냥 집계하면 아직 발생하지 않은 예약이 피처에 섞여 들어간다.
그것이 이 도메인 특유의 리드타임 누수다.

핵심 함수 :func:`observed_demand` 는 "as-of 시점까지 관측된 수요"만 센다.
모든 피처는 이 함수를 통해 만들어지므로 미래를 볼 수 없다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 관측 시점(as-of) = stay_date - HORIZON. 며칠 앞을 내다볼 것인가.
DEFAULT_HORIZON = 7
LAG_DAYS = (7, 14, 28)
ROLLING_WINDOWS = (7, 14, 28)


def observed_demand(bookings: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """as-of 시점까지 관측된, 숙소×숙박일 단위 예약 건수.

    ``as_of`` 가 ``None`` 이면 모든 예약을 센다(= 최종 수요).

    Args:
        bookings: ``property_id``, ``stay_date``, ``booked_at`` 컬럼을 가진 예약 원장.
        as_of: 관측 시점. 이 시각 **이후에 생성된 예약은 제외**한다.
    """
    df = bookings
    if as_of is not None:
        df = df[df["booked_at"] <= as_of]
    out = (
        df.groupby(["property_id", "stay_date"], observed=True)
        .size()
        .rename("demand")
        .reset_index()
    )
    return out


def final_demand_panel(bookings: pd.DataFrame, properties: pd.DataFrame) -> pd.DataFrame:
    """숙소 × 숙박일 전체 격자에 최종 수요를 채운 패널.

    예약이 0건인 날도 행으로 남긴다. 수요 0인 날을 빼면 지표가 왜곡된다.
    """
    dates = pd.date_range(bookings["stay_date"].min(), bookings["stay_date"].max(), freq="D")
    grid = pd.MultiIndex.from_product(
        [properties["property_id"], dates], names=["property_id", "stay_date"]
    ).to_frame(index=False)

    actual = observed_demand(bookings, as_of=None)
    panel = grid.merge(actual, on=["property_id", "stay_date"], how="left")
    panel["demand"] = panel["demand"].fillna(0).astype(int)
    return panel.sort_values(["property_id", "stay_date"]).reset_index(drop=True)


def _pickup(bookings: pd.DataFrame, panel: pd.DataFrame, horizon: int) -> pd.Series:
    """as-of 시점까지 이미 들어온 예약 수 (booking pace).

    호텔 수요 예측에서 가장 강한 신호다. as-of 로직이 없으면 만들 수 없는 피처이기도 하다.
    각 숙박일 d 에 대해 ``booked_at <= d - horizon`` 인 예약만 센다.
    """
    b = bookings.copy()
    b["_offset"] = (b["stay_date"] - b["booked_at"]).dt.days
    observed = b[b["_offset"] >= horizon]
    counts = (
        observed.groupby(["property_id", "stay_date"], observed=True)
        .size()
        .rename("pickup")
        .reset_index()
    )
    merged = panel.merge(counts, on=["property_id", "stay_date"], how="left")
    return merged["pickup"].fillna(0).astype(int)


def build_features(
    bookings: pd.DataFrame,
    properties: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """학습·예측에 쓰는 피처 테이블을 만든다.

    모든 시계열 피처는 **예측 시점(stay_date - horizon) 기준**으로 계산한다.

    - lag_k : k일 전 숙박일의 최종 수요. ``k >= horizon`` 이므로 예측 시점에 이미 확정된 값이다
    - rolling_mean_w : lag 계열의 이동 평균 (shift 후 계산해 자기 자신을 포함하지 않는다)
    - pickup : 예측 시점까지 이미 들어온 예약 수
    """
    if horizon < 1:
        raise ValueError("horizon 은 1 이상이어야 한다")
    if min(LAG_DAYS) < horizon:
        raise ValueError(
            f"lag({min(LAG_DAYS)})가 horizon({horizon})보다 작으면 예측 시점에 "
            "확정되지 않은 값을 참조하게 된다"
        )

    panel = final_demand_panel(bookings, properties)
    panel["pickup"] = _pickup(bookings, panel, horizon)

    g = panel.groupby("property_id", observed=True)["demand"]
    for k in LAG_DAYS:
        panel[f"lag_{k}"] = g.shift(k)
    for w in ROLLING_WINDOWS:
        # horizon 만큼 shift 한 뒤 평균 → 예측 시점 이후 값이 섞이지 않는다
        panel[f"rolling_mean_{w}"] = (
            g.shift(horizon).rolling(w, min_periods=max(2, w // 2)).mean()
        )
    panel["rolling_std_7"] = g.shift(horizon).rolling(7, min_periods=3).std()

    panel = _add_calendar(panel)
    panel = panel.merge(
        properties[["property_id", "region", "property_type", "capacity", "rating"]],
        on="property_id",
        how="left",
    )
    panel["horizon"] = horizon
    panel["as_of"] = panel["stay_date"] - pd.Timedelta(days=horizon)
    return panel


def _add_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel["stay_date"].dt
    panel["dow"] = d.dayofweek
    panel["is_weekend"] = d.dayofweek.isin((4, 5)).astype(int)
    panel["month"] = d.month
    panel["week_of_year"] = d.isocalendar().week.astype(int)
    return panel


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """모델 입력으로 쓸 수치형 컬럼."""
    cols = ["pickup", "rolling_std_7", "dow", "is_weekend", "month",
            "week_of_year", "capacity", "rating"]
    cols += [f"lag_{k}" for k in LAG_DAYS]
    cols += [f"rolling_mean_{w}" for w in ROLLING_WINDOWS]
    return [c for c in cols if c in panel.columns]


def drop_warmup(panel: pd.DataFrame) -> pd.DataFrame:
    """lag 이 채워지지 않은 초기 구간을 제거한다."""
    need = [f"lag_{k}" for k in LAG_DAYS]
    return panel.dropna(subset=need).reset_index(drop=True)


__all__ = [
    "DEFAULT_HORIZON", "LAG_DAYS", "ROLLING_WINDOWS",
    "observed_demand", "final_demand_panel", "build_features",
    "feature_columns", "drop_warmup",
]
