"""파이프라인을 한 번만 돌리고 붙들고 있는다.

패널 생성은 1초, 모델 한 번 학습은 3초쯤 걸린다. 요청마다 하면 못 쓴다.
그래서 **처음 부를 때 만들고 그다음부터는 재사용**한다.

모델 비교(6종 × 5폴드)는 분 단위라 여기서 계산하지 않는다.
그건 ``scripts/run_baseline.py`` 가 만들어 ``reports/`` 에 커밋해 둔 **측정 결과**를
읽어서 준다. 요청마다 다시 재는 숫자가 아니라 한 번 재고 기록한 숫자다.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd

from src.data.generator import GeneratorConfig, generate
from src.evaluation import walkforward as wf
from src.features.asof import DEFAULT_HORIZON, build_features, drop_warmup
from src.models.baseline import BASELINES
from src.models.learned import LEARNED

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# 예측에 쓰는 모델. reports/model_comparison.csv 에서 가장 성적이 좋았던 것.
SERVING_MODEL = "poisson_gbdt"


class Runtime:
    """패널과 예측을 지연 생성해 캐시한다."""

    def __init__(self) -> None:
        # RLock 이어야 한다. forecast 가 락을 잡은 채 panel 을 부르는데,
        # 일반 Lock 은 재진입이 안 돼서 같은 스레드가 자기 자신을 기다린다.
        self._lock = threading.RLock()
        self._panel: pd.DataFrame | None = None
        self._forecast: pd.DataFrame | None = None
        self._fold = None

    # ------------------------------------------------------------ 패널
    @property
    def panel(self) -> pd.DataFrame:
        if self._panel is None:
            with self._lock:
                if self._panel is None:
                    bookings, properties = generate(GeneratorConfig())
                    self._panel = drop_warmup(
                        build_features(bookings, properties, horizon=DEFAULT_HORIZON)
                    )
        return self._panel

    # ------------------------------------------------------------ 예측
    @property
    def forecast(self) -> pd.DataFrame:
        """가장 최근 폴드의 검증 구간을 예측한다.

        학습은 검증 시작보다 horizon 만큼 앞에서 끝난다. 그 공백이 없으면 예측 시점에
        확정되지 않은 값을 학습에 넣게 되고, 그게 이 저장소가 막으려는 누수다.
        """
        if self._forecast is None:
            with self._lock:
                if self._forecast is None:
                    folds = wf.make_folds(
                        self.panel["stay_date"], n_splits=5,
                        horizon=DEFAULT_HORIZON, valid_days=28,
                    )
                    fold = folds[-1]
                    train, valid = wf.split(self.panel, fold)
                    pred = LEARNED[SERVING_MODEL](train, valid)
                    out = valid.copy()
                    out["pred"] = pred.to_numpy()
                    self._forecast = out
                    self._fold = fold
        return self._forecast

    @property
    def fold(self):
        self.forecast  # 폴드는 예측과 함께 정해진다
        return self._fold

    # ------------------------------------------------------------ 측정 결과
    def report(self, name: str) -> pd.DataFrame | None:
        path = REPORTS / f"{name}.csv"
        return pd.read_csv(path) if path.exists() else None

    @property
    def models(self) -> list[str]:
        return sorted({*BASELINES, *LEARNED})


runtime = Runtime()

__all__ = ["DEFAULT_HORIZON", "Runtime", "SERVING_MODEL", "runtime"]
