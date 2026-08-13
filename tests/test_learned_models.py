"""학습 모델 검증.

성능이 아니라 **계약**을 고정한다. 예측이 카운트로서 말이 되는가,
전처리가 검증 구간을 훔쳐보지 않는가, 기준선을 실제로 이기는가.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.generator import GeneratorConfig, generate
from src.evaluation import walkforward as wf
from src.evaluation.metrics import wape
from src.features.asof import DEFAULT_HORIZON, build_features, drop_warmup
from src.models.baseline import seasonal_naive
from src.models.learned import LEARNED, _design, poisson_gbdt, poisson_glm


@pytest.fixture(scope="module")
def panel():
    cfg = GeneratorConfig(start="2024-01-01", end="2025-06-30", seed=11)
    bookings, properties = generate(cfg)
    return drop_warmup(build_features(bookings, properties, horizon=DEFAULT_HORIZON))


@pytest.fixture(scope="module")
def split(panel):
    folds = wf.make_folds(panel["stay_date"], n_splits=2,
                          horizon=DEFAULT_HORIZON, valid_days=28)
    return wf.split(panel, folds[-1])


# ------------------------------------------------------------ 예측의 성질
@pytest.mark.parametrize("name", sorted(LEARNED))
def test_predictions_are_valid_counts(split, name):
    """수요는 음수일 수 없고 결측일 수 없다."""
    train, valid = split
    pred = LEARNED[name](train, valid)
    assert len(pred) == len(valid)
    assert pred.notna().all(), f"{name}: NaN 예측"
    assert (pred >= 0).all(), f"{name}: 음수 예측"


@pytest.mark.parametrize("name", sorted(LEARNED))
def test_beats_seasonal_naive(split, name):
    """기준선을 못 이기면 채택하지 않는다 — 그 규칙을 테스트로 고정한다."""
    train, valid = split
    base = wape(valid["demand"], seasonal_naive(train, valid))
    got = wape(valid["demand"], LEARNED[name](train, valid))
    assert got < base, f"{name}: {got:.4f} 가 기준선 {base:.4f} 보다 나쁘다"


# ------------------------------------------------------------ 누수 방지
def test_design_matrix_imputes_from_train_only(split):
    """결측 대치는 학습 구간 통계로만 한다.

    검증 구간의 중앙값을 쓰면 미래를 훔쳐보는 것이고, 시계열에서는 그게 곧 누수다.
    검증 구간 값을 통째로 바꿔도 학습 설계행렬은 흔들리지 않아야 한다.
    """
    train, valid = split
    xtr_a, _ = _design(train, valid)

    poisoned = valid.copy()
    num = [c for c in poisoned.columns if poisoned[c].dtype.kind in "fi"]
    poisoned[num] = poisoned[num] * 1000 + 999

    xtr_b, _ = _design(train, poisoned)
    np.testing.assert_allclose(xtr_a, xtr_b)


def test_unseen_category_does_not_break_prediction(split):
    """학습 구간에 없던 범주가 와도 예측이 되어야 한다."""
    train, valid = split
    v = valid.copy()
    v.loc[v.index[:5], "region"] = "Atlantis"
    pred = poisson_glm(train, v)
    assert pred.notna().all()
    assert (pred >= 0).all()


def test_design_columns_match_between_train_and_valid(split):
    """설계행렬의 열 구성이 두 구간에서 같아야 한다."""
    train, valid = split
    xtr, xva = _design(train, valid)
    assert xtr.shape[1] == xva.shape[1]
    assert np.isfinite(xtr).all() and np.isfinite(xva).all()


# ------------------------------------------------------------ 재현성
def test_gbdt_is_deterministic(split):
    """같은 입력이면 같은 예측 — 비교가 성립하려면 필요하다."""
    train, valid = split
    a = poisson_gbdt(train, valid)
    b = poisson_gbdt(train, valid)
    pd.testing.assert_series_equal(a, b)
