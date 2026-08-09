"""walk-forward 검증.

일반 Random Split 을 쓰지 않는다. 시간 순서를 유지하고,
학습 구간과 검증 구간 사이에 **예측 horizon 만큼 gap** 을 둔다.

    Past ──────────────────────────────────────→ Future

    Train              gap        Validation
    ████████████████   ░░░░░      ██████
                       ↑
                 예측 리드타임만큼 비움

gap 이 없으면 학습 구간 끝과 검증 구간 시작이 맞닿아, 예측 시점에는 알 수 없는
정보가 학습에 들어간다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import pandas as pd

from .metrics import evaluate


@dataclass(frozen=True)
class Fold:
    """한 개 폴드의 시간 경계."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp

    @property
    def gap_days(self) -> int:
        return (self.valid_start - self.train_end).days - 1

    def describe(self) -> str:
        return (
            f"fold {self.index}  "
            f"train {self.train_start:%Y-%m-%d}~{self.train_end:%Y-%m-%d}  "
            f"gap {self.gap_days}d  "
            f"valid {self.valid_start:%Y-%m-%d}~{self.valid_end:%Y-%m-%d}"
        )


def make_folds(
    dates: pd.Series | pd.DatetimeIndex,
    n_splits: int = 5,
    horizon: int = 7,
    valid_days: int = 28,
) -> list[Fold]:
    """확장 윈도(expanding window) 폴드를 만든다.

    검증 구간을 뒤에서부터 ``n_splits`` 개 잘라내고, 각 폴드의 학습 구간은
    검증 시작보다 ``horizon`` 일 앞에서 끝난다.
    """
    d = pd.DatetimeIndex(pd.Series(dates).unique()).sort_values()
    last = d.max()
    folds: list[Fold] = []

    for i in range(n_splits):
        valid_end = last - pd.Timedelta(days=valid_days * i)
        valid_start = valid_end - pd.Timedelta(days=valid_days - 1)
        train_end = valid_start - pd.Timedelta(days=horizon)
        if train_end <= d.min():
            break
        folds.append(
            Fold(
                index=n_splits - i,
                train_start=d.min(),
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )
        )
    return sorted(folds, key=lambda f: f.valid_start)


def split(panel: pd.DataFrame, fold: Fold,
          date_col: str = "stay_date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """폴드 경계로 학습·검증 데이터를 나눈다."""
    tr = panel[(panel[date_col] >= fold.train_start) & (panel[date_col] <= fold.train_end)]
    va = panel[(panel[date_col] >= fold.valid_start) & (panel[date_col] <= fold.valid_end)]
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def run(
    panel: pd.DataFrame,
    fit_predict: Callable[[pd.DataFrame, pd.DataFrame], pd.Series],
    folds: list[Fold],
    y_col: str = "demand",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """폴드별로 학습·예측하고 지표를 모은다.

    Args:
        fit_predict: ``(train, valid) -> 예측값`` 함수.

    Returns:
        (폴드별 지표, 예측이 붙은 검증 데이터 전체)
    """
    rows, preds = [], []
    for fold in folds:
        tr, va = split(panel, fold)
        if tr.empty or va.empty:
            continue
        pred = fit_predict(tr, va)
        va = va.assign(pred=pd.Series(pred).to_numpy())
        m = evaluate(va[y_col], va["pred"])
        m["fold"] = fold.index
        m["valid_start"] = fold.valid_start
        m["valid_end"] = fold.valid_end
        m["gap_days"] = fold.gap_days
        rows.append(m)
        preds.append(va)

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        cols = ["fold", "valid_start", "valid_end", "gap_days"] + [
            c for c in metrics.columns if c not in ("fold", "valid_start", "valid_end", "gap_days")
        ]
        metrics = metrics[cols]
    return metrics, (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame())


def summarize(metrics: pd.DataFrame) -> dict[str, float]:
    """폴드 평균과 **편차**를 함께 낸다. 평균만 보고하지 않는다."""
    if metrics.empty:
        return {}
    return {
        "wape_mean": float(metrics["wape"].mean()),
        "wape_std": float(metrics["wape"].std(ddof=0)),
        "wape_worst": float(metrics["wape"].max()),
        "mae_mean": float(metrics["mae"].mean()),
        "rmse_mean": float(metrics["rmse"].mean()),
        "folds": int(len(metrics)),
    }


__all__ = ["Fold", "make_folds", "split", "run", "summarize"]
