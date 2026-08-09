"""리드타임 누수 방지 테스트 — 이 저장소에서 가장 중요한 테스트.

as-of 피처는 "예측 시점까지 관측된 예약"만 봐야 한다.
예측 시점 이후에 예약이 더 들어와도 **피처 값이 변하면 안 된다.**

이 테스트가 깨지면 모델 성능이 아무리 좋아도 그 숫자는 믿을 수 없다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.generator import GeneratorConfig, generate
from src.features.asof import (
    DEFAULT_HORIZON,
    LAG_DAYS,
    build_features,
    observed_demand,
)


@pytest.fixture(scope="module")
def data():
    cfg = GeneratorConfig(start="2024-01-01", end="2024-06-30", seed=7)
    return generate(cfg)


def test_booking_never_after_stay(data):
    """예약은 숙박일 이후에 생성될 수 없다."""
    bookings, _ = data
    assert (bookings["booked_at"] <= bookings["stay_date"]).all()


def test_observed_demand_excludes_future_bookings(data):
    """as-of 이후 예약은 집계에서 빠진다."""
    bookings, _ = data
    as_of = pd.Timestamp("2024-04-01")

    obs = observed_demand(bookings, as_of=as_of)
    final = observed_demand(bookings, as_of=None)

    merged = obs.merge(final, on=["property_id", "stay_date"], suffixes=("_obs", "_fin"))
    # 관측치는 최종치를 넘을 수 없다
    assert (merged["demand_obs"] <= merged["demand_fin"]).all()

    # as_of 이후에만 예약이 들어온 미래 숙박일은 관측치가 더 작아야 한다
    future = merged[merged["stay_date"] > as_of + pd.Timedelta(days=30)]
    assert (future["demand_obs"] < future["demand_fin"]).any()


def test_features_unchanged_when_future_bookings_appear(data):
    """핵심 검증.

    예측 시점 이후에 생성된 예약을 추가로 넣어도 피처가 그대로여야 한다.
    미래 정보가 피처에 새어 들어가면 이 테스트가 실패한다.
    """
    bookings, properties = data
    cutoff = pd.Timestamp("2024-05-01")

    # 시나리오 A: 예측 시점(cutoff) 이후에 생성된 예약은 아직 존재하지 않는 세계
    known = bookings[bookings["booked_at"] <= cutoff]
    # 시나리오 B: 그 뒤로 예약이 더 들어온 세계 (= 전체 데이터)
    full = bookings

    fa = build_features(known, properties, horizon=DEFAULT_HORIZON)
    fb = build_features(full, properties, horizon=DEFAULT_HORIZON)

    # 예측 시점이 cutoff 이전인 행만 비교한다.
    # 이 행들의 피처는 cutoff 이후 예약과 무관해야 한다.
    key = ["property_id", "stay_date"]
    feat_cols = ["pickup"] + [f"lag_{k}" for k in LAG_DAYS]

    a = fa[fa["as_of"] <= cutoff].set_index(key)[feat_cols].sort_index()
    b = fb.set_index(key).loc[a.index, feat_cols].sort_index()

    assert not a.empty, "비교할 행이 없다 — 테스트 구간 설정을 확인할 것"
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_pickup_never_exceeds_final_demand(data):
    """예측 시점까지 들어온 예약 수가 최종 수요를 넘을 수 없다."""
    bookings, properties = data
    panel = build_features(bookings, properties, horizon=DEFAULT_HORIZON)
    assert (panel["pickup"] <= panel["demand"]).all()


def test_lag_must_not_be_shorter_than_horizon(data):
    """horizon 보다 짧은 lag 를 쓰려 하면 즉시 실패해야 한다.

    예측 시점에 확정되지 않은 값을 참조하게 되기 때문이다.
    """
    bookings, properties = data
    with pytest.raises(ValueError, match="horizon"):
        build_features(bookings, properties, horizon=min(LAG_DAYS) + 1)


def test_rolling_features_do_not_include_target_day(data):
    """이동 평균이 자기 자신(예측 대상일)을 포함하지 않는다."""
    bookings, properties = data
    panel = build_features(bookings, properties, horizon=DEFAULT_HORIZON)

    one = panel[panel["property_id"] == panel["property_id"].iloc[0]].reset_index(drop=True)
    row = one[one["rolling_mean_7"].notna()].iloc[-1]
    pos = one.index[one["stay_date"] == row["stay_date"]][0]

    # horizon 만큼 shift 한 뒤 7일 평균이므로, 사용된 원본 구간은 [pos-13, pos-7]
    window = one["demand"].iloc[pos - 13 : pos - 6]
    assert np.isclose(row["rolling_mean_7"], window.mean())
