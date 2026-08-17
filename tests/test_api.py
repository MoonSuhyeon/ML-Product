"""HTTP 표면 — 배치 파이프라인의 규칙이 조회 층에서도 유지되는가.

여기서 확인하는 것은 예측 정확도가 아니다. 그건 walk-forward 테스트가 본다.
**API 가 시점을 숨기지 않는가, 평균 뒤를 가리지 않는가** 를 본다.
"""
from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from api.server import app
from src.features.asof import DEFAULT_HORIZON


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_forecast_reports_the_as_of_date(client):
    """예측 시점이 응답에 있어야 한다.

    이 저장소의 주장이 "피처를 예측 시점 기준으로 재구성했다" 인데,
    그 시점을 안 주면 확인할 방법이 없다.
    """
    body = client.get("/forecast", params={"limit": 5}).json()
    assert body["as_of"]
    assert body["horizon_days"] == DEFAULT_HORIZON
    assert body["rows"]
    for row in body["rows"]:
        assert row["as_of"] < row["stay_date"], "as_of 가 숙박일보다 앞서야 한다"


def test_training_ends_a_horizon_before_the_window(client):
    """학습 종료와 검증 시작 사이에 horizon 만큼 공백이 있어야 한다.

    그 공백이 누수를 구조적으로 막는다. API 응답으로도 확인 가능해야 한다.
    """
    from datetime import date
    body = client.get("/forecast", params={"limit": 1}).json()
    as_of = date.fromisoformat(body["as_of"])
    start = date.fromisoformat(body["window_start"])
    assert start - as_of == timedelta(days=DEFAULT_HORIZON)


def test_predictions_are_valid_counts(client):
    """수요는 음수일 수 없다."""
    rows = client.get("/forecast", params={"limit": 200}).json()["rows"]
    assert all(r["predicted"] >= 0 for r in rows)


def test_region_filter_narrows_the_result(client):
    everything = client.get("/forecast", params={"limit": 1}).json()["count"]
    jeju = client.get("/forecast", params={"region": "Jeju", "limit": 1}).json()
    assert 0 < jeju["count"] < everything
    assert all(r["region"] == "Jeju" for r in
               client.get("/forecast", params={"region": "Jeju", "limit": 20}).json()["rows"])


def test_low_demand_is_sorted_and_below_threshold(client):
    """콘솔이 이걸 받아 콘텐츠 생성으로 넘긴다. 정렬과 임계값이 지켜져야 한다."""
    body = client.get("/forecast/low-demand", params={"threshold": 0.8, "limit": 30}).json()
    preds = [r["predicted"] for r in body["rows"]]
    assert all(p < 0.8 for p in preds)
    assert preds == sorted(preds)


def test_segments_expose_the_sparsity_gradient(client):
    """평균 뒤에 가려지는 지역 편차를 드러내야 한다."""
    body = client.get("/forecast/segments").json()
    assert body["dimension"] == "region", "기본 축은 지역이다"
    rows = body["rows"]
    assert len(rows) >= 3
    assert all(0 <= r["zero_ratio"] <= 1 for r in rows)
    # 오차가 가장 큰 지역이 수요 0 비중도 가장 높다 — 희소성 기울기
    worst = max(rows, key=lambda r: r["wape"])
    sparsest = max(rows, key=lambda r: r["zero_ratio"])
    assert worst["key"] == sparsest["key"]


def test_segments_can_be_cut_by_property_type(client):
    """축이 하나뿐이면 "평균 뒤를 본다"는 주장이 반쪽이다.

    지역만 볼 수 있으면 특정 숙소 유형에서만 무너지는 모델을 지역 평균이
    덮어 가린다.
    """
    body = client.get("/forecast/segments", params={"by": "property_type"}).json()
    assert body["dimension"] == "property_type"
    keys = {r["key"] for r in body["rows"]}
    assert len(keys) >= 3
    assert "Seoul" not in keys, "지역이 섞여 나오면 축이 안 바뀐 것이다"


def test_segments_reject_an_axis_that_is_not_in_the_panel(client):
    """임의 문자열을 groupby 에 그대로 넘기면 사용자가 KeyError 를 본다."""
    r = client.get("/forecast/segments", params={"by": "nonexistent"})
    assert r.status_code == 400
    assert "nonexistent" in r.json()["detail"]


def test_every_dimension_covers_the_same_rows(client):
    """축을 바꿔도 같은 예측을 쪼갠 것이어야 한다. 합이 다르면 필터가 섞인 것이다."""
    totals = []
    for by in ("region", "property_type"):
        rows = client.get("/forecast/segments", params={"by": by}).json()["rows"]
        totals.append(sum(r["n"] for r in rows))
    assert totals[0] == totals[1]


def test_metrics_say_where_the_numbers_came_from(client):
    """요청마다 재는 게 아니라는 사실을 응답이 숨기지 않아야 한다."""
    body = client.get("/metrics").json()
    assert "run_baseline" in body["measured_by"]
    assert body["baseline"] == "seasonal_naive"
    assert body["serving"] == "poisson_gbdt"


def test_metrics_rank_the_serving_model_first(client):
    """서빙 모델이 비교표에서 1위가 아니면 서빙 선택이 틀린 것이다."""
    body = client.get("/metrics").json()
    rows = body["rows"]
    assert rows[0]["model"] == body["serving"]
    assert rows[0]["vs_baseline_pct"] > 0
    assert all(r["folds"] == 5 for r in rows)
