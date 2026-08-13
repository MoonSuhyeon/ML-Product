"""학습 모델 — 카운트 우도를 쓰는 두 후보.

**왜 제곱오차가 아니라 Poisson 인가.**
타깃은 하루 예약 건수다. 작은 정수이고, 13% 는 0이다. 제곱오차 회귀는 이 분포를
가정하지 않는다. 음수를 예측할 수 있고, 0 근처와 성수기의 오차를 같은 저울로 잰다.
목적함수를 분포에 맞추는 것이 학습기를 키우는 것보다 먼저다.

두 후보를 같은 walk-forward 프레임에 태워 기준선과 비교한다.

- ``poisson_glm``  — 선형, 해석 가능, 계수가 그대로 곱셈 효과
- ``poisson_gbdt`` — 상호작용을 잡되 우도는 Poisson 유지

**희박 구간에서 나아지는지가 판정 기준이다.** 전체 WAPE 만 좋아지고 경주(수요 0이
30%)가 그대로면 채택하지 않는다. 오차가 몰려 있는 곳이 거기이기 때문이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from src.features.asof import feature_columns

# 범주형은 학습 구간에서 본 값만 쓴다. 검증 구간에 새 값이 오면 0 벡터가 된다.
CATEGORICAL = ("region", "property_type")


def _design(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """학습·검증 설계행렬. 결측 대치와 인코딩은 **학습 구간에서만** 학습한다."""
    num = [c for c in feature_columns(train) if c in valid.columns]

    med = train[num].median()
    xtr = train[num].fillna(med)
    xva = valid[num].fillna(med)

    for col in CATEGORICAL:
        if col not in train.columns:
            continue
        levels = sorted(train[col].dropna().unique())
        for lv in levels:
            xtr[f"{col}={lv}"] = (train[col] == lv).astype(float)
            xva[f"{col}={lv}"] = (valid[col] == lv).astype(float)

    xva = xva[xtr.columns]
    return xtr.to_numpy(dtype=float), xva.to_numpy(dtype=float)


def poisson_glm(train: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """Poisson GLM.

    스케일링은 GLM 에만 적용한다. 계수 추정이 수렴 속도와 정규화 강도에 영향을 받기
    때문이다. ``fit`` 은 학습 구간에서만 하고 검증에는 ``transform`` 만 쓴다 —
    시계열에서도 같은 원칙이다.
    """
    xtr, xva = _design(train, valid)
    scaler = StandardScaler().fit(xtr)
    model = PoissonRegressor(alpha=1e-3, max_iter=2000)
    model.fit(scaler.transform(xtr), train["demand"].to_numpy(dtype=float))
    pred = model.predict(scaler.transform(xva))
    return pd.Series(pred).clip(lower=0)


def poisson_gbdt(train: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
    """Poisson 손실을 쓰는 히스토그램 부스팅.

    트리는 스케일에 무관하므로 스케일링하지 않는다. 대출 과제에서 세운
    "LR 은 필수, RF 는 불필요" 판단과 같은 기준이다.
    """
    xtr, xva = _design(train, valid)
    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    model.fit(xtr, train["demand"].to_numpy(dtype=float))
    return pd.Series(model.predict(xva)).clip(lower=0)


LEARNED = {
    "poisson_glm": poisson_glm,
    "poisson_gbdt": poisson_gbdt,
}

__all__ = ["LEARNED", "poisson_glm", "poisson_gbdt"]
