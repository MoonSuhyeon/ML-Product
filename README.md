# As-of Feature Engineering for Demand Forecasting

*Personal project*

![Data / ML Engineering](https://img.shields.io/badge/Data%20%2F%20ML%20Engineering-0B1220?style=for-the-badge)

![temporal correctness](https://img.shields.io/badge/temporal%20correctness-1D4ED8?style=for-the-badge) ![feature engineering](https://img.shields.io/badge/feature%20engineering-1D4ED8?style=for-the-badge) ![validation](https://img.shields.io/badge/validation-B45309?style=for-the-badge) ![leakage prevention](https://img.shields.io/badge/leakage%20prevention-BE123C?style=for-the-badge)

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

**WAPE 0.3276** · **+50.1%** over the seasonal baseline · **24 tests**, leakage guards included

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

## Stack

| | |
|---|---|
| Language | Python 3.11 |
| Data | pandas · NumPy |
| Modeling | scikit-learn — `PoissonRegressor` · `HistGradientBoostingRegressor(loss='poisson')` |
| Testing | pytest — 24 tests, leakage guards included |
| Reports | CSV under `reports/` |

---

## Trade-offs

Measured on synthetic reservation data — 83,861 bookings · 41 properties ·
2 years · walk-forward 5 folds · horizon 7d · gap 6d.

### Batch prediction instead of real-time inference

Serving a forecast per request means assembling features per request. Demand for
a stay date does not change minute to minute, so that cost buys nothing.

**Buys** — serving becomes a cached lookup rather than feature assembly plus a
model call, and the model runs once per day instead of once per request.
**Costs** — same-day responsiveness. An event announcement or a wave of
cancellations is not reflected until the next batch.

### Matching the likelihood to the data, not scaling the learner

The target is a daily booking count — small integers, 13% of them zero. Squared
error does not assume that shape: it can predict negative demand and it weighs a
miss near zero the same as a miss in peak season. So the first move was the
objective, not the model size.

Six candidates ran through the same walk-forward frame.

| Model | WAPE | vs baseline |
|---|---|---|
| **Poisson GBDT** | **0.3276** | **+50.1%** |
| Poisson GLM | 0.3790 | +42.2% |
| Pickup Ratio *(rule)* | 0.4486 | +31.6% |
| Weekday Mean | 0.4908 | +25.2% |
| Moving Average | 0.5710 | +13.0% |
| Seasonal Naive *(baseline)* | 0.6562 | — |

**Buys** — a correctly specified likelihood is worth more than a bigger model
here. The linear Poisson GLM alone reaches +42.2%, ahead of every rule, and
adding interactions on top of the same likelihood takes it to +50.1%.
**Costs** — a model to train, version and watch for drift, where the rule had
none. Prediction is no longer something you can read off a spreadsheet.

The rule is kept in the comparison rather than deleted. It is what makes the
+50.1% mean something, and `pickup_ratio` still needs no training at all — a
useful fallback when a model cannot be served.

### Improvement was uniform, but the sparsity gradient survived

The acceptance test for the learned models was not the average. Error had been
concentrated in sparse-demand regions, so a model that only improved the busy
ones would not have been worth its operational weight.

| Region | zero-demand days | Pickup Ratio | **Poisson GBDT** | improvement |
|---|---|---|---|---|
| Seoul | 6.6% | 0.3759 | **0.2758** | 26.6% |
| Busan | 9.6% | 0.4239 | **0.3173** | 25.1% |
| Jeju | 15.2% | 0.4664 | **0.3448** | 26.1% |
| Gangneung | 23.5% | 0.6078 | **0.4125** | **32.1%** |
| Gyeongju | 30.1% | 0.6707 | **0.4892** | 27.1% |

**Buys** — every segment improved by 25–32%, and the largest gain landed on one
of the sparsest regions rather than the easiest one.
**Costs** — the ordering did not change. Gyeongju is still worst and still
sparsest. The Poisson likelihood lifted the whole curve; it did not flatten it.
Sparse-demand handling remains a separate problem, and the next thing worth
building — cold start, pooling across similar properties — is still the same
one it was before.

### WAPE as the primary metric, MAPE excluded

13.1% of property-days have zero demand, and 30.1% in the sparsest region. MAPE
divides by those.

**Buys** — a metric that stays finite and weights by volume, so a busy property
and a quiet one do not distort each other.
**Costs** — MAPE is easier to explain to a non-technical reader. That was traded
for a number that does not mislead.

### One global model instead of one per property

**Buys** — a single artifact to train, version and serve rather than 41, and
sparse properties borrow signal from the shared structure instead of fitting on
their own thin history. Region and property type enter as features, so the model
can still separate them.
**Costs** — no per-property tuning. A property with an unusual pattern is
averaged toward its neighbours, and the sparsity gradient above is the visible
form of that trade.

### A gap inside the validation split

**Buys** — leakage is prevented structurally, so the numbers above mean what they
claim. A test asserts that adding future bookings changes no feature value.
**Costs** — the gap removes training rows, and lag features shorter than the
horizon are rejected outright rather than silently used.

## Run locally

```bash
pip install -r requirements.txt
pytest                        # 24 tests
python scripts/run_baseline.py
```

## Docs

| | |
|---|---|
| `src/features/asof.py` | As-of featuring — the core of the project |
| `src/evaluation/walkforward.py` | Fold construction and the gap |
| `src/models/learned.py` | Poisson GLM and GBDT — why the likelihood, not the size |
| `tests/test_asof_leakage.py` | Proof that future bookings change nothing |
| `reports/` | Measured comparisons, per-fold and per-segment |
