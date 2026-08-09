"""기준선 모델.

**Baseline 우선.** Seasonal Naive 를 못 이기는 모델은 채택하지 않는다.
복잡한 모델을 먼저 붙이면 개선인지 착시인지 구분할 수 없다.

모든 모델은 ``(train, valid) -> 예측값`` 형태로 통일해
:func:`src.evaluation.walkforward.run` 에 그대로 넘길 수 있게 한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive(train: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """전주 같은 요일의 수요를 그대로 예측값으로 쓴다.

    ``lag_7`` 이 곧 전주 동일 요일 값이다. 요일 주기가 강한 도메인에서
    이 단순한 규칙이 상당히 강한 기준선이 된다.
    """
    pred = valid["lag_7"].astype(float)
    return pred.fillna(train["demand"].mean()).clip(lower=0)


def moving_average(train: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """최근 4주 이동 평균."""
    pred = valid["rolling_mean_28"].astype(float)
    return pred.fillna(train["demand"].mean()).clip(lower=0)


def weekday_mean(train: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """숙소 × 요일 평균. 학습 구간에서만 통계를 낸다."""
    stat = (
        train.groupby(["property_id", "dow"], observed=True)["demand"]
        .mean()
        .rename("pred")
        .reset_index()
    )
    merged = valid[["property_id", "dow"]].merge(stat, on=["property_id", "dow"], how="left")
    fallback = float(train["demand"].mean())
    return merged["pred"].fillna(fallback).clip(lower=0)


def pickup_ratio(train: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """예약 페이스(pickup)를 최종 수요로 환산한다.

    학습 구간에서 ``최종 수요 / pickup`` 배수를 요일별로 추정한 뒤,
    검증 구간의 pickup 에 곱한다. as-of 피처가 있어야만 가능한 기준선이다.
    """
    tr = train[train["pickup"] > 0]
    if tr.empty:
        return seasonal_naive(train, valid)

    ratio = (
        tr.assign(_r=tr["demand"] / tr["pickup"])
        .groupby("dow", observed=True)["_r"]
        .median()
        .rename("ratio")
        .reset_index()
    )
    overall = float((tr["demand"].sum() / tr["pickup"].sum()) or 1.0)

    m = valid[["dow", "pickup"]].merge(ratio, on="dow", how="left")
    m["ratio"] = m["ratio"].fillna(overall)
    pred = m["pickup"].astype(float) * m["ratio"]

    # pickup 이 0 이면 정보가 없으므로 계절 naive 로 대체
    fallback = seasonal_naive(train, valid).to_numpy()
    pred = pd.Series(np.where(m["pickup"].to_numpy() > 0, pred.to_numpy(), fallback))
    return pred.clip(lower=0)


BASELINES = {
    "seasonal_naive": seasonal_naive,
    "moving_average": moving_average,
    "weekday_mean": weekday_mean,
    "pickup_ratio": pickup_ratio,
}

__all__ = ["seasonal_naive", "moving_average", "weekday_mean", "pickup_ratio", "BASELINES"]
