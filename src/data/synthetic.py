"""
Synthetic M5-style retail data for tests, CI smoke runs and demos.

The generator produces the same three tables as the M5 competition (sales,
calendar, prices) with the same column names, so every code path that handles
real M5 files can be exercised without downloading anything.

Demand is simulated, not sampled from any real retailer. Each series has a
base rate drawn from a log-normal distribution (a mix of slow and fast movers),
multiplied by day-of-week and yearly seasonality, a small trend, event and
SNAP uplifts, and a price response (promotions raise demand). Daily sales are
then drawn from a Poisson distribution with gamma noise, which gives integer
counts and realistic runs of zeros for slow movers.

Because the data-generating process is known, the features the model should
find useful (lags, calendar, price, events) really do carry signal. That makes
the smoke run a meaningful check: a model that cannot beat a naive baseline on
this data is broken.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SyntheticDataConfig(BaseModel):
    """Configuration for synthetic data generation."""

    n_items: int = Field(default=100, ge=1, le=10000)
    n_stores: int = Field(default=4, ge=1, le=10, description="Up to the 10 M5 store IDs")
    n_days: int = Field(default=1941, ge=56, description="M5 has 1,941 days")
    start_date: str = Field(default="2011-01-29")

    # Demand
    base_demand_mean: float = Field(default=3.0, gt=0, description="Mean daily rate per series")
    base_demand_dispersion: float = Field(
        default=0.9, ge=0, description="Log-normal sigma of per-series base rates"
    )
    noise_shape: float = Field(default=4.0, gt=0, description="Gamma shape of daily noise")
    trend_per_year: float = Field(default=0.03, description="Relative demand growth per year")

    # Events and SNAP
    event_probability: float = Field(default=0.05, ge=0, le=1)
    event_uplift: float = Field(default=1.3, ge=1)
    snap_uplift_foods: float = Field(default=1.15, ge=1)

    # Price
    price_mean: float = Field(default=5.0, gt=0)
    price_std: float = Field(default=2.0, ge=0)
    promotion_probability: float = Field(default=0.1, ge=0, le=1)
    promotion_discount: float = Field(default=0.2, ge=0, lt=1)
    price_elasticity: float = Field(default=1.5, ge=0)

    random_seed: int = Field(default=42)


class SyntheticDataGenerator:
    """Generate synthetic M5-style sales, calendar and price tables.

    Example:
        >>> generator = SyntheticDataGenerator(SyntheticDataConfig(n_items=21, n_stores=2))
        >>> sales, calendar, prices = generator.generate_all()
    """

    STORES: dict[str, list[str]] = {  # noqa: RUF012 - read-only lookup table
        "CA": ["CA_1", "CA_2", "CA_3", "CA_4"],
        "TX": ["TX_1", "TX_2", "TX_3"],
        "WI": ["WI_1", "WI_2", "WI_3"],
    }

    DEPARTMENTS: list[tuple[str, str]] = [  # noqa: RUF012 - read-only lookup table
        ("FOODS", "FOODS_1"),
        ("FOODS", "FOODS_2"),
        ("FOODS", "FOODS_3"),
        ("HOBBIES", "HOBBIES_1"),
        ("HOBBIES", "HOBBIES_2"),
        ("HOUSEHOLD", "HOUSEHOLD_1"),
        ("HOUSEHOLD", "HOUSEHOLD_2"),
    ]

    EVENTS: list[tuple[str, str]] = [  # noqa: RUF012 - read-only lookup table
        ("SuperBowl", "Sporting"),
        ("ValentinesDay", "Cultural"),
        ("PresidentsDay", "National"),
        ("StPatricksDay", "Cultural"),
        ("Easter", "Religious"),
        ("MemorialDay", "National"),
        ("IndependenceDay", "National"),
        ("LaborDay", "National"),
        ("Halloween", "Cultural"),
        ("Thanksgiving", "National"),
        ("Christmas", "Religious"),
        ("NewYear", "National"),
    ]

    # Ten SNAP-like benefit days per month for each state (illustrative).
    SNAP_DAYS: dict[str, set[int]] = {  # noqa: RUF012 - read-only lookup table
        "CA": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        "TX": {1, 3, 5, 6, 7, 9, 11, 12, 13, 15},
        "WI": {2, 3, 5, 6, 8, 9, 11, 12, 14, 15},
    }

    # Monday .. Sunday multipliers: weekend peak.
    DAY_OF_WEEK_EFFECT = np.array([0.90, 0.85, 0.85, 0.90, 1.05, 1.25, 1.20])

    def __init__(self, data_config: SyntheticDataConfig | None = None) -> None:
        self.data_config = data_config or SyntheticDataConfig()

    # ------------------------------------------------------------------ public

    def generate_all(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Generate (sales, calendar, prices), consistent with each other."""
        rng = np.random.default_rng(self.data_config.random_seed)
        calendar = self._calendar(rng)
        items, stores = self._items(), self._stores()
        prices, price_matrix, regular = self._prices(rng, calendar, items, stores)
        sales = self._sales(rng, calendar, items, stores, price_matrix, regular)
        logger.info(
            "Generated synthetic data: %s sales rows, %s series, %s days",
            f"{len(sales):,}",
            f"{len(items) * len(stores):,}",
            len(calendar),
        )
        return sales, calendar, prices

    def generate_calendar(self) -> pl.DataFrame:
        """Generate only the calendar table."""
        return self.generate_all()[1]

    def generate_sales(self) -> pl.DataFrame:
        """Generate only the (long-format) sales table."""
        return self.generate_all()[0]

    def generate_prices(self) -> pl.DataFrame:
        """Generate only the weekly price table."""
        return self.generate_all()[2]

    def write_m5_files(self, out_dir: str | Path) -> dict[str, Path]:
        """Write the synthetic tables as M5-format CSVs (wide sales)."""
        from src.data.loader import write_m5_files

        sales, calendar, prices = self.generate_all()
        return write_m5_files(sales, calendar, prices, out_dir)

    # ----------------------------------------------------------------- helpers

    def _items(self) -> list[tuple[str, str, str]]:
        """Return (item_id, dept_id, cat_id), spreading items across departments."""
        items = []
        for i in range(self.data_config.n_items):
            cat_id, dept_id = self.DEPARTMENTS[i % len(self.DEPARTMENTS)]
            items.append((f"{dept_id}_{i + 1:03d}", dept_id, cat_id))
        return items

    def _stores(self) -> list[str]:
        all_stores = [s for stores in self.STORES.values() for s in stores]
        return all_stores[: self.data_config.n_stores]

    def _dates(self) -> list[date]:
        start = date.fromisoformat(self.data_config.start_date)
        return [start + timedelta(days=i) for i in range(self.data_config.n_days)]

    def _calendar(self, rng: np.random.Generator) -> pl.DataFrame:
        dates = self._dates()
        n = len(dates)
        is_event = rng.random(n) < self.data_config.event_probability
        event_idx = rng.integers(0, len(self.EVENTS), n)
        event_name = [
            self.EVENTS[j][0] if e else None for e, j in zip(is_event, event_idx, strict=True)
        ]
        event_type = [
            self.EVENTS[j][1] if e else None for e, j in zip(is_event, event_idx, strict=True)
        ]

        data: dict[str, list[object]] = {
            "date": list(dates),
            "wm_yr_wk": [d.isocalendar()[0] * 100 + d.isocalendar()[1] for d in dates],
            "weekday": [d.strftime("%A") for d in dates],
            "wday": [d.isoweekday() for d in dates],
            "month": [d.month for d in dates],
            "year": [d.year for d in dates],
            "d": [f"d_{i + 1}" for i in range(n)],
            "event_name_1": list(event_name),
            "event_type_1": list(event_type),
            "event_name_2": [None] * n,
            "event_type_2": [None] * n,
        }
        for state, days in self.SNAP_DAYS.items():
            data[f"snap_{state}"] = [int(d.day in days) for d in dates]

        return pl.DataFrame(
            data,
            schema_overrides={
                "event_name_1": pl.String,
                "event_type_1": pl.String,
                "event_name_2": pl.String,
                "event_type_2": pl.String,
            },
        )

    def _prices(
        self,
        rng: np.random.Generator,
        calendar: pl.DataFrame,
        items: list[tuple[str, str, str]],
        stores: list[str],
    ) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
        """Weekly prices per series, plus (series x day) price and regular price arrays."""
        cfg = self.data_config
        weeks = calendar["wm_yr_wk"].unique(maintain_order=True).to_numpy()
        n_series, n_weeks = len(items) * len(stores), len(weeks)

        base = np.maximum(0.5, rng.normal(cfg.price_mean, cfg.price_std, len(items)))
        regular = np.repeat(base, len(stores)) * (0.95 + 0.1 * rng.random(n_series))
        promo = rng.random((n_series, n_weeks)) < cfg.promotion_probability
        wobble = 0.98 + 0.04 * rng.random((n_series, n_weeks))
        weekly = np.round(np.where(promo, 1 - cfg.promotion_discount, wobble) * regular[:, None], 2)

        prices = pl.DataFrame(
            {
                "store_id": np.repeat([s for _ in items for s in stores], n_weeks),
                "item_id": np.repeat([it[0] for it in items for _ in stores], n_weeks),
                "wm_yr_wk": np.tile(weeks, n_series),
                "sell_price": weekly.ravel(),
            }
        )

        week_pos = {w: i for i, w in enumerate(weeks)}
        day_week_idx = np.array([week_pos[w] for w in calendar["wm_yr_wk"].to_list()])
        return prices, weekly[:, day_week_idx], regular

    def _sales(
        self,
        rng: np.random.Generator,
        calendar: pl.DataFrame,
        items: list[tuple[str, str, str]],
        stores: list[str],
        price_matrix: np.ndarray,
        regular: np.ndarray,
    ) -> pl.DataFrame:
        cfg = self.data_config
        n_days = len(calendar)
        n_series = len(items) * len(stores)
        dates = calendar["date"].to_list()
        t = np.arange(n_days)

        dow = np.array([d.weekday() for d in dates])
        doy = np.array([d.timetuple().tm_yday for d in dates])
        seasonal = (
            self.DAY_OF_WEEK_EFFECT[dow]
            * (1 + 0.15 * np.sin(2 * np.pi * (doy - 258) / 365.25))
            * (1 + cfg.trend_per_year * t / 365.25)
        )
        event = np.where(calendar["event_name_1"].is_not_null().to_numpy(), cfg.event_uplift, 1.0)

        sigma = cfg.base_demand_dispersion
        base = np.exp(rng.normal(np.log(cfg.base_demand_mean) - sigma**2 / 2, sigma, len(items)))
        store_mult = 0.8 + 0.4 * rng.random(n_series)
        series_base = np.repeat(base, len(stores)) * store_mult

        states = [s.split("_")[0] for _ in items for s in stores]
        is_food = np.array([it[2] == "FOODS" for it in items for _ in stores])
        snap = np.vstack([calendar[f"snap_{st}"].to_numpy() for st in states])
        snap_mult = np.where((snap == 1) & is_food[:, None], cfg.snap_uplift_foods, 1.0)

        price_effect = (price_matrix / regular[:, None]) ** (-cfg.price_elasticity)
        noise = rng.gamma(cfg.noise_shape, 1 / cfg.noise_shape, (n_series, n_days))

        rate = series_base[:, None] * seasonal[None, :] * event[None, :]
        counts = rng.poisson(rate * snap_mult * price_effect * noise)

        ids = [f"{it[0]}_{s}_evaluation" for it in items for s in stores]
        return pl.DataFrame(
            {
                "id": np.repeat(ids, n_days),
                "item_id": np.repeat([it[0] for it in items for _ in stores], n_days),
                "dept_id": np.repeat([it[1] for it in items for _ in stores], n_days),
                "cat_id": np.repeat([it[2] for it in items for _ in stores], n_days),
                "store_id": np.repeat([s for _ in items for s in stores], n_days),
                "state_id": np.repeat(states, n_days),
                "d": np.tile(calendar["d"].to_numpy(), n_series),
                "sales": counts.ravel().astype(np.int64),
            }
        )
