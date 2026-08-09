"""Phase 3 파이프라인 — 데이터 생성 → 피처 → 기준선 → walk-forward 검증.

    python scripts/run_baseline.py

결과는 ``reports/`` 에 CSV 로 저장하고 요약을 표준출력으로 낸다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 콘솔 기본 인코딩(Windows cp949)에서도 한글·기호가 깨지지 않게 한다
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.generator import GeneratorConfig, generate          # noqa: E402
from src.evaluation import walkforward as wf                      # noqa: E402
from src.evaluation.metrics import evaluate_by                    # noqa: E402
from src.features.asof import DEFAULT_HORIZON, build_features, drop_warmup  # noqa: E402
from src.models.baseline import BASELINES                         # noqa: E402

REPORTS = ROOT / "reports"
HORIZON = DEFAULT_HORIZON


def main() -> int:
    pd.set_option("display.width", 120)
    REPORTS.mkdir(exist_ok=True)

    print("=" * 72)
    print("Phase 0  숙박 예약 데이터 생성")
    bookings, properties = generate(GeneratorConfig())
    print(f"  예약 {len(bookings):,}건 · 숙소 {len(properties)}개 · "
          f"{bookings['stay_date'].min():%Y-%m-%d} ~ {bookings['stay_date'].max():%Y-%m-%d}")
    print(f"  평균 리드타임 {bookings['lead_time'].mean():.1f}일 "
          f"(중앙값 {bookings['lead_time'].median():.0f}일)")

    print()
    print("=" * 72)
    print(f"Phase 2  as-of 피처 생성 (horizon={HORIZON}일)")
    panel = build_features(bookings, properties, horizon=HORIZON)
    panel = drop_warmup(panel)
    zero = float((panel["demand"] == 0).mean())
    print(f"  패널 {len(panel):,}행 · 수요 0인 날 비중 {zero:.1%}")
    print(f"  → 수요 0 비중이 높아 MAPE 를 주지표로 쓸 수 없다. WAPE 를 사용한다.")

    print()
    print("=" * 72)
    print(f"Phase 3  walk-forward 검증 (gap={HORIZON - 1}일)")
    folds = wf.make_folds(panel["stay_date"], n_splits=5, horizon=HORIZON, valid_days=28)
    for f in folds:
        print("  " + f.describe())

    print()
    print("=" * 72)
    print("기준선 비교")
    summary_rows = []
    all_metrics = {}
    best_preds = None
    best_wape = float("inf")

    for name, fn in BASELINES.items():
        metrics, preds = wf.run(panel, fn, folds)
        s = wf.summarize(metrics)
        s["model"] = name
        summary_rows.append(s)
        all_metrics[name] = metrics
        if s["wape_mean"] < best_wape:
            best_wape, best_preds, best_name = s["wape_mean"], preds, name

    summary = pd.DataFrame(summary_rows)
    summary = summary[["model", "wape_mean", "wape_std", "wape_worst",
                       "mae_mean", "rmse_mean", "folds"]]
    summary = summary.sort_values("wape_mean").reset_index(drop=True)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ref = summary[summary["model"] == "seasonal_naive"]["wape_mean"].iloc[0]
    print()
    print(f"  기준선(seasonal_naive) WAPE = {ref:.4f}")
    for r in summary.itertuples(index=False):
        if r.model == "seasonal_naive":
            continue
        delta = (ref - r.wape_mean) / ref * 100
        verdict = "개선" if delta > 0 else "악화"
        print(f"  {r.model:<16} {r.wape_mean:.4f}  기준선 대비 {delta:+.1f}% {verdict}")

    print()
    print("=" * 72)
    print(f"폴드별 편차 — {best_name} (평균만 보고하지 않는다)")
    fm = all_metrics[best_name][["fold", "valid_start", "valid_end", "wape", "mae", "n"]]
    print(fm.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print()
    print("=" * 72)
    print(f"세그먼트별 오차 — {best_name}")
    by_region = evaluate_by(best_preds, "region")
    print(by_region[["region", "wape", "mae", "zero_ratio", "n"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    by_month = evaluate_by(best_preds, "month")
    worst = by_month.sort_values("wape", ascending=False).head(3)
    print()
    print("  오차가 큰 상위 3개월:")
    print(worst[["month", "wape", "mae", "n"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    summary.to_csv(REPORTS / "model_comparison.csv", index=False)
    all_metrics[best_name].to_csv(REPORTS / "folds_best.csv", index=False)
    by_region.to_csv(REPORTS / "error_by_region.csv", index=False)
    print()
    print(f"저장: {REPORTS.relative_to(ROOT)}/  (model_comparison · folds_best · error_by_region)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
