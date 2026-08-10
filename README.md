# Demand Forecasting Without Leaking the Future

*Personal project*

In a hospitality marketplace, **hosts have to commit to prices and inventory
before demand shows up.** Getting that wrong costs money in both directions —
empty rooms on a peak weekend, or a sold-out property priced too low.

Forecasting looks like a standard time series problem until you notice that
**bookings arrive before the stay.** Demand for August 15th is not a number that
exists on August 15th; it accumulates for weeks beforehand. So the honest
question is not "what will demand be" but **"what can I know about August 15th
on August 1st, using only what has arrived by then?"**

If you aggregate by stay date and stop there, bookings that have not been made
yet end up inside your features. The model looks excellent and fails in
production. That is lead-time leakage, and it is specific to this domain.

I built a forecasting pipeline where **every feature is reconstructed as of the
prediction moment, validation keeps a gap the size of the forecast horizon, and
a test proves that adding future bookings does not change a single feature
value.**

---

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │           RESERVATION LEDGER         │
                    │                                      │
                    │  booking_id · property_id · region   │
                    │  stay_date  ·  booked_at  · price    │
                    │                                      │
                    │  booked_at ≤ stay_date  (invariant)  │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │          PANEL BUILDER               │
                    │                                      │
                    │  property × stay_date full grid      │
                    │  zero-demand days kept, not dropped  │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
        ┌──────────────────────────────────────────────────────────┐
        │                  AS-OF FEATURE STORE                     │
        │              as_of = stay_date − horizon                 │
        │                                                          │
        │  ┌────────────────┐  ┌────────────────┐  ┌────────────┐  │
        │  │ PICKUP         │  │ LAG            │  │ ROLLING    │  │
        │  │ bookings       │  │ lag_7 · 14     │  │ mean 7·14  │  │
        │  │ already in     │  │ lag_28         │  │ ·28 · std  │  │
        │  │ at as_of       │  │ (k ≥ horizon)  │  │ shift-then │  │
        │  └────────────────┘  └────────────────┘  └────────────┘  │
        │                                                          │
        │  ┌────────────────┐  ┌────────────────┐  ┌────────────┐  │
        │  │ CALENDAR       │  │ PROPERTY       │  │ PRICE      │  │
        │  │ dow · weekend  │  │ region · type  │  │ level      │  │
        │  │ month · week   │  │ capacity·rating│  │ rel. to    │  │
        │  └────────────────┘  └────────────────┘  │ regional μ │  │
        │                                          └────────────┘  │
        │                                                          │
        │  guard: lag_k < horizon  →  ValueError                    │
        └──────────────────────────┬───────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
    ┌────────────────────┐                ┌─────────────────────────┐
    │      MODELS        │                │  WALK-FORWARD SPLIT     │
    │                    │                │                         │
    │ Seasonal Naive ◀───┼── reference    │  expanding window       │
    │ Moving Average     │                │  5 folds × 28d valid    │
    │ Weekday Mean       │                │                         │
    │ Pickup Ratio       │                │  Train ██████ gap ░░    │
    │  (needs as-of)     │                │              Valid ███  │
    └─────────┬──────────┘                │  gap = horizon          │
              │                           └────────────┬────────────┘
              └────────────────┬───────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────────────────┐
        │                     EVALUATION                           │
        │                                                          │
        │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
        │  │ WAPE  primary│  │ MAE · RMSE   │  │ MAPE  excluded │  │
        │  │ Σ|y−ŷ| / Σ|y|│  │ secondary    │  │ diverges on    │  │
        │  │              │  │              │  │ zero-demand    │  │
        │  └──────────────┘  └──────────────┘  └────────────────┘  │
        │                                                          │
        │  per-fold variance · per-region · per-month              │
        │  reports/  model_comparison · folds · error_by_region    │
        └──────────────────────────────────────────────────────────┘

        ┌──────────────────────────────────────────────────────────┐
        │                    LEAKAGE TESTS                         │
        │                                                          │
        │  features(bookings ≤ cutoff) == features(all bookings)   │
        │  pickup ≤ final demand · rolling excludes target day     │
        │  folds never overlap · gap always applied                │
        └──────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────────┐
                    │        SERVING  (planned)            │
                    │  FastAPI /forecast · quantile band   │
                    │  drift monitor → retrain trigger     │
                    └──────────────────────────────────────┘
```

---

## Results

Synthetic reservation data — 83,861 bookings · 41 properties · 2 years.
Walk-forward, 5 folds, horizon 7d, gap 6d.

| Model | WAPE | vs baseline |
|---|---|---|
| **Pickup Ratio** | **0.4486** | **+31.6%** |
| Weekday Mean | 0.4908 | +25.2% |
| Moving Average | 0.5710 | +13.0% |
| Seasonal Naive *(baseline)* | 0.6562 | — |

The strongest signal is **pickup** — how many bookings already exist at the
prediction moment. It is the one feature that cannot be built without as-of
logic, which is why the pipeline was built around it first.

Error is not uniform. It tracks sparsity:

| Region | WAPE | zero-demand days |
|---|---|---|
| Seoul | 0.376 | 6.6% |
| Jeju | 0.466 | 15.2% |
| Gyeongju | 0.671 | 30.1% |

Reporting only the 0.449 average hides Gyeongju. Sparse-demand handling is the
next thing worth building, not a stronger learner.

## Stack

| | |
|---|---|
| Language | Python 3.11 |
| Data | pandas · NumPy |
| Modeling | scikit-learn (Poisson-family and GBDT planned) |
| Testing | pytest — 16 tests, leakage guards included |
| Reports | CSV under `reports/` |

## Run locally

```bash
pip install -r requirements.txt
pytest                        # 16 tests
python scripts/run_baseline.py
```

## Docs

| | |
|---|---|
| `src/features/asof.py` | As-of featuring — the core of the project |
| `src/evaluation/walkforward.py` | Fold construction and the gap |
| `tests/test_asof_leakage.py` | Proof that future bookings change nothing |
| `reports/` | Measured comparisons, per-fold and per-segment |
