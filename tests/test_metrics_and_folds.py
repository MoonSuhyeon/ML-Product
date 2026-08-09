"""지표 계산과 폴드 분할 검증."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import evaluate, mae, mape, rmse, wape, zero_ratio
from src.evaluation.walkforward import make_folds, split


# ---------------------------------------------------------------- metrics
def test_perfect_prediction_is_zero_error():
    y = [3, 0, 5, 2]
    assert wape(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0


def test_wape_matches_definition():
    y = [10, 0, 5]
    p = [8, 1, 5]
    # |10-8| + |0-1| + |5-5| = 3,  분모 = 15
    assert np.isclose(wape(y, p), 3 / 15)


def test_wape_survives_zero_actuals():
    """수요 0 이 섞여도 발산하지 않는다. MAPE 를 배제한 이유."""
    y = [0, 0, 4]
    p = [1, 0, 3]
    assert np.isfinite(wape(y, p))


def test_mape_excludes_zero_actuals():
    y = [0, 4]
    p = [2, 2]
    # 0 인 실측은 제외 → |4-2|/4 = 0.5
    assert np.isclose(mape(y, p), 0.5)


def test_zero_ratio_is_reported():
    assert zero_ratio([0, 0, 1, 3]) == 0.5


def test_evaluate_returns_all_keys():
    m = evaluate([1, 2, 0], [1, 1, 1])
    for k in ("wape", "mae", "rmse", "mape_nonzero", "zero_ratio", "n"):
        assert k in m
    assert m["n"] == 3


# ------------------------------------------------------------------ folds
@pytest.fixture
def dates():
    return pd.date_range("2024-01-01", "2024-12-31", freq="D")


def test_folds_have_requested_gap(dates):
    horizon = 7
    folds = make_folds(dates, n_splits=5, horizon=horizon, valid_days=28)
    assert folds
    for f in folds:
        assert f.gap_days == horizon - 1, f.describe()


def test_folds_never_overlap_train_and_valid(dates):
    folds = make_folds(dates, n_splits=5, horizon=7, valid_days=28)
    for f in folds:
        assert f.train_end < f.valid_start


def test_folds_move_forward_in_time(dates):
    folds = make_folds(dates, n_splits=4, horizon=7, valid_days=28)
    starts = [f.valid_start for f in folds]
    assert starts == sorted(starts)


def test_split_respects_boundaries(dates):
    panel = pd.DataFrame({"stay_date": dates, "demand": 1})
    fold = make_folds(dates, n_splits=3, horizon=7, valid_days=28)[0]
    tr, va = split(panel, fold)
    assert tr["stay_date"].max() <= fold.train_end
    assert va["stay_date"].min() >= fold.valid_start
    # 학습 데이터에 검증 구간 날짜가 단 하나도 없어야 한다
    assert set(tr["stay_date"]).isdisjoint(set(va["stay_date"]))
