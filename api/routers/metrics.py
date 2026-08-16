"""모델 비교 — 요청마다 재는 게 아니라 재어 둔 것을 준다.

6종 × 5폴드는 분 단위라 요청 경로에 둘 수 없다. ``scripts/run_baseline.py`` 가
측정해 ``reports/model_comparison.csv`` 에 커밋해 둔 결과를 서빙한다.
**응답에 measured_by 를 실어 그 사실을 숨기지 않는다.**
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import MetricsResponse, ModelRow
from api.state import SERVING_MODEL, runtime

router = APIRouter(tags=["metrics"])

BASELINE = "seasonal_naive"


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    df = runtime.report("model_comparison")
    if df is None:
        raise HTTPException(
            503, "측정 결과가 없습니다. python scripts/run_baseline.py 를 먼저 실행하세요"
        )

    base = df.loc[df["model"] == BASELINE, "wape_mean"]
    ref = float(base.iloc[0]) if len(base) else None

    rows = []
    for r in df.sort_values("wape_mean").itertuples(index=False):
        gain = None
        if ref and r.model != BASELINE:
            gain = round((ref - r.wape_mean) / ref * 100, 1)
        rows.append(ModelRow(
            model=r.model,
            wape_mean=round(float(r.wape_mean), 4),
            wape_std=round(float(r.wape_std), 4),
            wape_worst=round(float(r.wape_worst), 4),
            mae_mean=round(float(r.mae_mean), 4),
            folds=int(r.folds),
            vs_baseline_pct=gain,
        ))

    return MetricsResponse(
        baseline=BASELINE,
        serving=SERVING_MODEL,
        measured_by="scripts/run_baseline.py — walk-forward 5폴드, reports/ 에 커밋됨",
        rows=rows,
    )
