"""예측 오차 지표.

숙소 단위로 내려가면 예약 0건인 날이 흔하다. 그래서 MAPE 는 발산한다.
전체 예약량을 분모로 쓰는 **WAPE 를 주지표**로 둔다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _to_array(x) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.to_numpy()
    return np.asarray(x, dtype=float).ravel()


def mae(y_true, y_pred) -> float:
    y, p = _to_array(y_true), _to_array(y_pred)
    return float(np.mean(np.abs(y - p)))


def rmse(y_true, y_pred) -> float:
    y, p = _to_array(y_true), _to_array(y_pred)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error.

    ``sum|y - yhat| / sum|y|``. 분모가 전체 합이라 개별 0 값에 발산하지 않는다.
    """
    y, p = _to_array(y_true), _to_array(y_pred)
    denom = np.sum(np.abs(y))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y - p)) / denom)


def mape(y_true, y_pred) -> float:
    """참고용. **0 인 실측을 제외**하고 계산한다.

    제외한다는 사실 자체가 이 지표를 주지표로 쓸 수 없는 이유다.
    """
    y, p = _to_array(y_true), _to_array(y_pred)
    mask = y != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])))


def zero_ratio(y_true) -> float:
    """실측이 0 인 비율. MAPE 를 배제한 근거를 수치로 남긴다."""
    y = _to_array(y_true)
    return float(np.mean(y == 0))


def evaluate(y_true, y_pred) -> dict[str, float]:
    """주지표와 보조 지표를 한 번에."""
    return {
        "wape": wape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape_nonzero": mape(y_true, y_pred),
        "zero_ratio": zero_ratio(y_true),
        "n": int(len(_to_array(y_true))),
    }


def evaluate_by(df: pd.DataFrame, group: str,
                y_col: str = "demand", p_col: str = "pred") -> pd.DataFrame:
    """세그먼트별 오차.

    전체 WAPE 가 좋아도 특정 지역·성수기에서 무너지면 실패로 본다.
    """
    rows = []
    for key, g in df.groupby(group, observed=True):
        m = evaluate(g[y_col], g[p_col])
        m[group] = key
        rows.append(m)
    out = pd.DataFrame(rows)
    cols = [group] + [c for c in out.columns if c != group]
    return out[cols].sort_values("wape").reset_index(drop=True)


__all__ = ["mae", "rmse", "wape", "mape", "zero_ratio", "evaluate", "evaluate_by"]
