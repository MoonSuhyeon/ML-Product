"""숙박 예약 더미 데이터 생성기.

실제 예약 데이터의 핵심 성질을 재현한다.

- 예약은 **숙박일보다 먼저** 발생한다 (리드타임)
- 요일·계절·지역·숙소에 따라 수요가 다르다
- 가격이 수요에 영향을 준다

리드타임이 있기 때문에 "숙박일 기준 집계"와 "예약 시점 기준 관측"이 달라지고,
이 차이가 as-of 피처링이 필요한 이유가 된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 지역별 특성: (숙소 수, 기본 일일 수요, 성수기 월)
REGIONS: dict[str, tuple[int, float, tuple[int, ...]]] = {
    "Seoul": (12, 3.2, (5, 10)),
    "Busan": (8, 2.6, (7, 8)),
    "Jeju": (10, 2.9, (7, 8)),
    "Gangneung": (6, 1.9, (7, 8)),
    "Gyeongju": (5, 1.6, (4, 10)),
}

PROPERTY_TYPES = ("APARTMENT", "HOTEL", "GUESTHOUSE", "PENSION", "HOUSE")

# 요일별 수요 배수 (월=0 … 일=6). 금·토가 높다.
WEEKDAY_FACTOR = np.array([0.72, 0.68, 0.75, 0.88, 1.45, 1.70, 0.95])


@dataclass
class GeneratorConfig:
    """생성 파라미터."""

    start: str = "2024-01-01"
    end: str = "2025-12-31"
    seed: int = 42
    # 리드타임 분포: 평균 약 12일, 꼬리가 긴 형태
    lead_time_mean: float = 12.0
    lead_time_max: int = 120
    # 가격 탄력성 (지역 평균 대비 비율에 대한 지수)
    price_elasticity: float = -0.9
    holidays: tuple[str, ...] = field(
        default_factory=lambda: (
            "2024-01-01", "2024-02-09", "2024-02-10", "2024-02-12", "2024-03-01",
            "2024-05-05", "2024-05-15", "2024-06-06", "2024-08-15", "2024-09-16",
            "2024-09-17", "2024-09-18", "2024-10-03", "2024-10-09", "2024-12-25",
            "2025-01-01", "2025-01-28", "2025-01-29", "2025-01-30", "2025-03-01",
            "2025-05-05", "2025-05-06", "2025-06-06", "2025-08-15", "2025-10-03",
            "2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09", "2025-12-25",
        )
    )


def build_properties(cfg: GeneratorConfig) -> pd.DataFrame:
    """숙소 마스터를 만든다."""
    rng = np.random.default_rng(cfg.seed)
    rows = []
    pid = 1
    for region, (n_props, base_demand, peak_months) in REGIONS.items():
        for _ in range(n_props):
            ptype = PROPERTY_TYPES[rng.integers(len(PROPERTY_TYPES))]
            rows.append(
                {
                    "property_id": f"P{pid:03d}",
                    "region": region,
                    "property_type": ptype,
                    "capacity": int(rng.integers(2, 9)),
                    "rating": round(float(rng.uniform(3.6, 4.9)), 1),
                    "review_count": int(rng.integers(5, 800)),
                    # 숙소마다 인기도가 다르다
                    "popularity": round(float(rng.lognormal(0.0, 0.45)), 3),
                    "base_price": int(rng.integers(7, 26)) * 10_000,
                    "base_demand": base_demand,
                    "peak_months": peak_months,
                }
            )
            pid += 1
    return pd.DataFrame(rows)


def _season_factor(dates: pd.DatetimeIndex, peak_months: tuple[int, ...]) -> np.ndarray:
    """성수기 가산 + 완만한 연간 사이클."""
    month = dates.month.to_numpy()
    peak = np.isin(month, peak_months).astype(float) * 0.55
    cycle = 0.12 * np.sin(2 * np.pi * (dates.dayofyear.to_numpy() / 365.25))
    return 1.0 + peak + cycle


def _price_series(rng: np.random.Generator, base_price: int, n: int,
                  season: np.ndarray) -> np.ndarray:
    """성수기에 오르고 잡음이 섞인 가격."""
    noise = rng.normal(1.0, 0.06, size=n)
    return np.round(base_price * (0.85 + 0.35 * season) * noise, -3)


def generate(cfg: GeneratorConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """예약 원장과 숙소 마스터를 만든다.

    Returns:
        bookings: 예약 1건이 1행. ``booked_at`` 은 항상 ``stay_date`` 이전이다.
        properties: 숙소 마스터.
    """
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)

    properties = build_properties(cfg)
    dates = pd.date_range(cfg.start, cfg.end, freq="D")
    holidays = set(pd.to_datetime(list(cfg.holidays)))
    n_days = len(dates)

    weekday = WEEKDAY_FACTOR[dates.dayofweek.to_numpy()]
    is_holiday = np.array([d in holidays for d in dates], dtype=float)

    booking_rows: list[dict] = []
    booking_id = 1

    for prop in properties.itertuples(index=False):
        season = _season_factor(dates, prop.peak_months)
        price = _price_series(rng, prop.base_price, n_days, season)

        # 지역 평균 대비 가격 → 수요 배수
        rel_price = price / np.mean(price)
        price_factor = rel_price ** cfg.price_elasticity

        lam = (
            prop.base_demand
            * prop.popularity
            * weekday
            * season
            * price_factor
            * (1.0 + 0.35 * is_holiday)
        )
        # 평점이 높을수록 소폭 유리
        lam = lam * (0.85 + 0.06 * (prop.rating - 3.5))
        demand = rng.poisson(np.clip(lam, 0.02, None))

        for i, stay_date in enumerate(dates):
            k = int(demand[i])
            if k == 0:
                continue
            # 리드타임: 성수기일수록 더 일찍 예약한다
            scale = cfg.lead_time_mean * (0.8 + 0.5 * (season[i] - 1.0))
            lead = rng.exponential(max(scale, 2.0), size=k).astype(int) + 1
            lead = np.clip(lead, 1, cfg.lead_time_max)
            for j in range(k):
                booking_rows.append(
                    {
                        "booking_id": booking_id,
                        "property_id": prop.property_id,
                        "region": prop.region,
                        "stay_date": stay_date,
                        "booked_at": stay_date - pd.Timedelta(days=int(lead[j])),
                        "lead_time": int(lead[j]),
                        "price": float(price[i]),
                    }
                )
                booking_id += 1

    bookings = pd.DataFrame(booking_rows)
    bookings = bookings.sort_values(["stay_date", "booked_at"]).reset_index(drop=True)

    properties = properties.drop(columns=["base_demand", "peak_months"])
    return bookings, properties


__all__ = ["GeneratorConfig", "generate", "build_properties", "REGIONS"]
