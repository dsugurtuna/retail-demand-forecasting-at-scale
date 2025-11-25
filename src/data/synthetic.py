"""
Synthetic data generation for testing and development.

Generates realistic M5-like data for testing the forecasting pipeline
without requiring access to the actual dataset.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.utils.config import Config

logger = logging.getLogger(__name__)


class SyntheticDataConfig(BaseModel):
    """Configuration for synthetic data generation."""
    
    n_items: int = Field(default=100, ge=1, le=10000)
    n_stores: int = Field(default=4, ge=1, le=100)
    n_days: int = Field(default=1941, ge=28)  # M5 has 1941 days
    start_date: str = Field(default="2011-01-29")
    
    # Demand parameters
    base_demand_mean: float = Field(default=5.0, ge=0)
    base_demand_std: float = Field(default=2.0, ge=0)
    
    # Seasonality
    weekly_seasonality: bool = Field(default=True)
    monthly_seasonality: bool = Field(default=True)
    yearly_seasonality: bool = Field(default=True)
    
    # Events
    event_probability: float = Field(default=0.05, ge=0, le=1)
    event_uplift: float = Field(default=1.5, ge=1)
    
    # Price
    price_mean: float = Field(default=5.0, gt=0)
    price_std: float = Field(default=2.0, ge=0)
    promotion_probability: float = Field(default=0.1, ge=0, le=1)
    promotion_discount: float = Field(default=0.2, ge=0, le=1)
    
    random_seed: int = Field(default=42)


class SyntheticDataGenerator:
    """
    Generate synthetic retail data for testing and development.
    
    Creates realistic M5-like data with:
    - Multiple hierarchies (item, category, store, state)
    - Temporal patterns (weekly, monthly, yearly seasonality)
    - Events and holidays
    - Price variations and promotions
    - SNAP (food assistance) effects
    
    Example:
        >>> generator = SyntheticDataGenerator()
        >>> sales = generator.generate_sales()
        >>> calendar = generator.generate_calendar()
        >>> prices = generator.generate_prices()
    """
    
    STORES = {
        "CA": ["CA_1", "CA_2", "CA_3", "CA_4"],
        "TX": ["TX_1", "TX_2", "TX_3"],
        "WI": ["WI_1", "WI_2", "WI_3"],
    }
    
    CATEGORIES = {
        "FOODS": ["FOODS_1", "FOODS_2", "FOODS_3"],
        "HOBBIES": ["HOBBIES_1", "HOBBIES_2"],
        "HOUSEHOLD": ["HOUSEHOLD_1", "HOUSEHOLD_2"],
    }
    
    EVENTS = [
        ("SuperBowl", "Sporting"),
        ("ValentinesDay", "Cultural"),
        ("PresidentsDay", "National"),
        ("LentStart", "Religious"),
        ("LentWeek2", "Religious"),
        ("StPatricksDay", "Cultural"),
        ("Purim End", "Religious"),
        ("OrthodoxEaster", "Religious"),
        ("Pesach End", "Religious"),
        ("Cinco De Mayo", "Cultural"),
        ("Mother's day", "Cultural"),
        ("MemorialDay", "National"),
        ("NBAFinalsStart", "Sporting"),
        ("NBAFinalsEnd", "Sporting"),
        ("Father's day", "Cultural"),
        ("IndependenceDay", "National"),
        ("Ramadan starts", "Religious"),
        ("Eid al-Fitr", "Religious"),
        ("LaborDay", "National"),
        ("ColumbusDay", "National"),
        ("Halloween", "Cultural"),
        ("EidAlAdha", "Religious"),
        ("VeteransDay", "National"),
        ("Thanksgiving", "National"),
        ("Christmas", "Religious"),
        ("Chanukah End", "Religious"),
        ("NewYear", "National"),
        ("OrthodoxChristmas", "Religious"),
        ("MartinLutherKingDay", "National"),
        ("Easter", "Religious"),
    ]
    
    def __init__(
        self, 
        config: Config | None = None,
        data_config: SyntheticDataConfig | None = None
    ) -> None:
        """
        Initialize generator.
        
        Args:
            config: Main application config
            data_config: Synthetic data specific config
        """
        self.data_config = data_config or SyntheticDataConfig()
        self._rng = np.random.default_rng(self.data_config.random_seed)
        
        # Extract settings from main config if provided
        if config is not None and hasattr(config, "data"):
            if hasattr(config.data, "start_date"):
                self.data_config.start_date = config.data.start_date
    
    def generate_all(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Generate all synthetic datasets.
        
        Returns:
            Tuple of (sales, calendar, prices) DataFrames
        """
        logger.info("Generating synthetic data...")
        calendar = self.generate_calendar()
        sales = self.generate_sales()
        prices = self.generate_prices()
        
        logger.info(
            f"Generated synthetic data: "
            f"{len(sales):,} sales rows, "
            f"{len(calendar):,} calendar rows, "
            f"{len(prices):,} price rows"
        )
        
        return sales, calendar, prices
    
    def generate_calendar(self) -> pl.DataFrame:
        """Generate calendar DataFrame with dates, events, and SNAP flags."""
        start_date = datetime.strptime(self.data_config.start_date, "%Y-%m-%d")
        dates = [start_date + timedelta(days=i) for i in range(self.data_config.n_days)]
        
        # Generate base calendar
        calendar_data = {
            "date": dates,
            "d": [f"d_{i+1}" for i in range(len(dates))],
            "wm_yr_wk": [d.isocalendar()[0] * 100 + d.isocalendar()[1] for d in dates],
            "weekday": [d.strftime("%A") for d in dates],
            "wday": [d.isoweekday() for d in dates],
            "month": [d.month for d in dates],
            "year": [d.year for d in dates],
        }
        
        # Add events (sparse)
        event_mask = self._rng.random(len(dates)) < self.data_config.event_probability
        event_names_1 = []
        event_types_1 = []
        
        for is_event in event_mask:
            if is_event:
                event = self._rng.choice(len(self.EVENTS))
                event_names_1.append(self.EVENTS[event][0])
                event_types_1.append(self.EVENTS[event][1])
            else:
                event_names_1.append(None)
                event_types_1.append(None)
        
        calendar_data["event_name_1"] = event_names_1
        calendar_data["event_type_1"] = event_types_1
        calendar_data["event_name_2"] = [None] * len(dates)
        calendar_data["event_type_2"] = [None] * len(dates)
        
        # Add SNAP flags (semi-random pattern)
        for state in self.STORES.keys():
            snap_pattern = self._generate_snap_pattern(len(dates))
            calendar_data[f"snap_{state}"] = snap_pattern
        
        return pl.DataFrame(calendar_data)
    
    def generate_sales(self) -> pl.DataFrame:
        """Generate sales DataFrame with realistic demand patterns."""
        # Generate item and store combinations
        items = self._generate_items()
        stores = self._get_all_stores()
        
        n_items = min(len(items), self.data_config.n_items)
        n_stores = min(len(stores), self.data_config.n_stores)
        
        items = items[:n_items]
        stores = stores[:n_stores]
        
        sales_records = []
        
        for item in items:
            item_base_demand = max(
                0,
                self._rng.normal(
                    self.data_config.base_demand_mean,
                    self.data_config.base_demand_std
                )
            )
            
            for store in stores:
                store_multiplier = 0.8 + self._rng.random() * 0.4  # 0.8 to 1.2
                
                # Generate daily sales
                daily_sales = self._generate_demand_series(
                    item_base_demand * store_multiplier,
                    self.data_config.n_days
                )
                
                for day_idx, sales in enumerate(daily_sales):
                    sales_records.append({
                        "id": f"{item['item_id']}_{store}_evaluation",
                        "item_id": item["item_id"],
                        "dept_id": item["dept_id"],
                        "cat_id": item["cat_id"],
                        "store_id": store,
                        "state_id": store.split("_")[0],
                        "d": f"d_{day_idx + 1}",
                        "sales": max(0, int(round(sales))),
                    })
        
        return pl.DataFrame(sales_records)
    
    def generate_prices(self) -> pl.DataFrame:
        """Generate price DataFrame with variations and promotions."""
        items = self._generate_items()[:self.data_config.n_items]
        stores = self._get_all_stores()[:self.data_config.n_stores]
        
        # Get unique weeks
        start_date = datetime.strptime(self.data_config.start_date, "%Y-%m-%d")
        weeks = set()
        for day in range(self.data_config.n_days):
            d = start_date + timedelta(days=day)
            weeks.add(d.isocalendar()[0] * 100 + d.isocalendar()[1])
        weeks = sorted(weeks)
        
        price_records = []
        
        for item in items:
            # Base price for item
            base_price = max(
                0.5,
                self._rng.normal(
                    self.data_config.price_mean,
                    self.data_config.price_std
                )
            )
            
            for store in stores:
                # Store-specific price variation (±5%)
                store_price = base_price * (0.95 + self._rng.random() * 0.1)
                
                for week in weeks:
                    # Possible promotion
                    if self._rng.random() < self.data_config.promotion_probability:
                        price = store_price * (1 - self.data_config.promotion_discount)
                    else:
                        # Small weekly variation
                        price = store_price * (0.98 + self._rng.random() * 0.04)
                    
                    price_records.append({
                        "store_id": store,
                        "item_id": item["item_id"],
                        "wm_yr_wk": week,
                        "sell_price": round(price, 2),
                    })
        
        return pl.DataFrame(price_records)
    
    def _generate_items(self) -> list[dict]:
        """Generate item metadata."""
        items = []
        item_counter = 1
        
        for cat_id, dept_ids in self.CATEGORIES.items():
            for dept_id in dept_ids:
                # Generate items per department
                n_items_per_dept = max(1, self.data_config.n_items // 7)
                
                for _ in range(n_items_per_dept):
                    items.append({
                        "item_id": f"{dept_id}_{item_counter:03d}",
                        "dept_id": dept_id,
                        "cat_id": cat_id,
                    })
                    item_counter += 1
        
        return items
    
    def _get_all_stores(self) -> list[str]:
        """Get flat list of all stores."""
        stores = []
        for state_stores in self.STORES.values():
            stores.extend(state_stores)
        return stores
    
    def _generate_demand_series(self, base_demand: float, n_days: int) -> np.ndarray:
        """Generate demand series with seasonality and noise."""
        t = np.arange(n_days)
        demand = np.ones(n_days) * base_demand
        
        # Weekly seasonality (higher on weekends)
        if self.data_config.weekly_seasonality:
            weekly = 0.3 * np.sin(2 * np.pi * t / 7)
            demand = demand * (1 + weekly)
        
        # Monthly seasonality
        if self.data_config.monthly_seasonality:
            monthly = 0.15 * np.sin(2 * np.pi * t / 30.4)
            demand = demand * (1 + monthly)
        
        # Yearly seasonality (higher in Q4)
        if self.data_config.yearly_seasonality:
            yearly = 0.2 * np.sin(2 * np.pi * (t - 90) / 365)
            demand = demand * (1 + yearly)
        
        # Add noise
        noise = self._rng.normal(0, base_demand * 0.3, n_days)
        demand = demand + noise
        
        # Add trend (slight growth)
        trend = 1 + 0.0001 * t
        demand = demand * trend
        
        return np.maximum(0, demand)
    
    def _generate_snap_pattern(self, n_days: int) -> list[int]:
        """Generate SNAP benefit pattern."""
        pattern = []
        for day in range(n_days):
            # SNAP benefits typically available at start of month
            day_of_month = (day % 30) + 1
            is_snap_day = day_of_month <= 10  # First 10 days of month
            pattern.append(int(is_snap_day))
        return pattern
