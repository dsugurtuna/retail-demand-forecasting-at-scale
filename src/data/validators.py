"""
Data validation with Pandera schemas plus a few plain-Python business rules.

Validation runs before feature engineering. The training entry point stops on
any error, because a model trained on malformed data produces confident
numbers that mean nothing.

What counts as an error (blocks training):
- a required column is missing or has the wrong type,
- a value breaks a schema check (for example negative sales),
- a required column is more than ``null_threshold`` null,
- duplicate (series, day) rows, or less history than ``min_history_days``.

What counts as a warning (logged, does not block):
- nulls in optional columns such as event names or prices before launch,
- a high share of statistical outliers in sales.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import pandera.pandas as pa
import polars as pl
from pandera.typing import Series
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class SalesDataSchema(pa.DataFrameModel):
    """Schema for long-format sales: one row per series per day."""

    id: Series[str] = pa.Field(nullable=False, description="Item-store series identifier")
    item_id: Series[str] = pa.Field(nullable=False)
    store_id: Series[str] = pa.Field(nullable=False)
    d: Series[str] | None = pa.Field(nullable=False, description="M5 day label (d_1, d_2, ...)")
    date: Series[pd.Timestamp] = pa.Field(nullable=False)
    sales: Series[float] = pa.Field(ge=0, nullable=True, description="Units sold, non-negative")

    class Config:
        coerce = True
        strict = False  # extra columns (calendar, prices) are allowed


class CalendarDataSchema(pa.DataFrameModel):
    """Schema for the M5 calendar table."""

    d: Series[str] = pa.Field(nullable=False)
    date: Series[pd.Timestamp] = pa.Field(nullable=False)
    wm_yr_wk: Series[int] = pa.Field(ge=0, nullable=False)
    weekday: Series[str] = pa.Field(nullable=False)
    wday: Series[int] = pa.Field(ge=1, le=7, nullable=False)
    month: Series[int] = pa.Field(ge=1, le=12, nullable=False)
    year: Series[int] = pa.Field(ge=2000, le=2100, nullable=False)

    class Config:
        coerce = True
        strict = False


class PriceDataSchema(pa.DataFrameModel):
    """Schema for the M5 weekly price table."""

    store_id: Series[str] = pa.Field(nullable=False)
    item_id: Series[str] = pa.Field(nullable=False)
    wm_yr_wk: Series[int] = pa.Field(ge=0, nullable=False)
    sell_price: Series[float] = pa.Field(gt=0, nullable=True, description="Positive price")

    class Config:
        coerce = True
        strict = False


class ValidationResult(BaseModel):
    """Outcome of a validation run."""

    is_valid: bool = Field(..., description="True when there are no errors")
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    row_count: int = Field(default=0)
    null_percentages: dict[str, float] = Field(default_factory=dict)

    @field_validator("errors", "warnings", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> list[str]:
        return [] if v is None else list(v)


SCHEMAS: dict[str, type[pa.DataFrameModel]] = {
    "sales": SalesDataSchema,
    "calendar": CalendarDataSchema,
    "prices": PriceDataSchema,
}


class DataValidator:
    """Validate sales, calendar or price data before training.

    Example:
        >>> result = DataValidator().validate(df, "sales")
        >>> if not result.is_valid:
        ...     print(result.errors)
    """

    def __init__(
        self,
        null_threshold: float = 0.05,
        outlier_std_threshold: float = 4.0,
        min_history_days: int = 28,
    ) -> None:
        self.null_threshold = null_threshold
        self.outlier_std_threshold = outlier_std_threshold
        self.min_history_days = min_history_days

    def validate(
        self,
        data: pl.DataFrame | pd.DataFrame,
        schema_name: str = "sales",
        raise_on_error: bool = False,
    ) -> ValidationResult:
        """Validate ``data`` against a named schema and the business rules."""
        if schema_name not in SCHEMAS:
            raise ValueError(f"Unknown schema '{schema_name}'. Choose from {sorted(SCHEMAS)}")

        df = data.to_pandas() if isinstance(data, pl.DataFrame) else data
        errors = self._validate_schema(df, schema_name)
        warnings: list[str] = []

        # Null thresholds only bind on columns the schema requires. Optional
        # columns (event names, prices before an item launches) are often
        # mostly null by design, so they are reported but do not block.
        required = set(SCHEMAS[schema_name].to_schema().columns)
        null_percentages = self._null_percentages(df)
        for col, pct in null_percentages.items():
            if pct <= self.null_threshold:
                continue
            message = f"Column '{col}' is {pct:.1%} null (threshold {self.null_threshold:.1%})"
            if col in required:
                errors.append(message)
            else:
                warnings.append(message)

        warnings.extend(self._validate_statistics(df))
        errors.extend(self._validate_business_rules(df))

        result = ValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
            row_count=len(df),
            null_percentages=null_percentages,
        )

        for warning in warnings:
            logger.warning("Validation warning: %s", warning)
        if errors:
            for error in errors:
                logger.error("Validation error: %s", error)
            if raise_on_error:
                raise ValueError(f"Data validation failed: {errors}")
        else:
            logger.info("Validation passed for %s rows (%s)", f"{len(df):,}", schema_name)
        return result

    def _validate_schema(self, df: pd.DataFrame, schema_name: str) -> list[str]:
        try:
            SCHEMAS[schema_name].validate(df, lazy=True)
        except pa.errors.SchemaErrors as exc:
            cases = exc.failure_cases[["column", "check", "failure_case"]].astype(str)
            missing = cases[cases["check"] == "column_in_dataframe"]
            other = cases[cases["check"] != "column_in_dataframe"]
            errors = [
                f"Missing required column '{name}'"
                for name in missing["failure_case"].drop_duplicates()
            ]
            errors += [
                f"Schema error in column '{row.column}': {row.check}"
                for row in other.drop_duplicates(["column", "check"]).itertuples(index=False)
            ]
            return errors
        return []

    @staticmethod
    def _null_percentages(df: pd.DataFrame) -> dict[str, float]:
        if len(df) == 0:
            return {}
        return {str(k): float(v) for k, v in (df.isna().sum() / len(df)).items()}

    def _validate_statistics(self, df: pd.DataFrame) -> list[str]:
        if "sales" not in df.columns:
            return []
        sales = pd.to_numeric(df["sales"], errors="coerce")
        std = sales.std()
        if not std or pd.isna(std):
            return []
        outlier_share = float(
            ((sales - sales.mean()).abs() > self.outlier_std_threshold * std).mean()
        )
        if outlier_share > 0.01:
            return [
                f"'sales' has {outlier_share:.1%} values more than "
                f"{self.outlier_std_threshold} standard deviations from the mean"
            ]
        return []

    def _validate_business_rules(self, df: pd.DataFrame) -> list[str]:
        errors = []

        if "date" in df.columns and len(df) > 0:
            dates = pd.to_datetime(df["date"])
            span_days = (dates.max() - dates.min()).days
            if span_days < self.min_history_days:
                errors.append(
                    f"Insufficient history: {span_days} days, minimum {self.min_history_days}"
                )

        key = ["id", "date"] if {"id", "date"}.issubset(df.columns) else ["id", "d"]
        if set(key).issubset(df.columns):
            duplicates = int(df.duplicated(subset=key).sum())
            if duplicates:
                errors.append(f"Found {duplicates} duplicate {'/'.join(key)} rows")

        if "sales" in df.columns:
            negative = int((pd.to_numeric(df["sales"], errors="coerce") < 0).sum())
            if negative:
                errors.append(f"Found {negative} negative sales values")

        return errors

    def validate_features(self, features: pd.DataFrame, target: str = "sales") -> ValidationResult:
        """Check a feature matrix for infinities, constant columns and near-duplicates."""
        errors: list[str] = []
        warnings: list[str] = []
        numeric_cols = list(features.select_dtypes(include=["number"]).columns)

        for col in numeric_cols:
            inf_count = int(features[col].isin([float("inf"), float("-inf")]).sum())
            if inf_count:
                errors.append(f"Column '{col}' contains {inf_count} infinite values")
            if col != target and features[col].std() == 0:
                warnings.append(f"Column '{col}' has zero variance")

        if len(numeric_cols) > 1:
            corr = features[numeric_cols].corr().to_numpy(dtype=float)
            for i, col1 in enumerate(numeric_cols):
                for j in range(i + 1, len(numeric_cols)):
                    col2, value = numeric_cols[j], float(corr[i, j])
                    if abs(value) > 0.99:
                        warnings.append(
                            f"High correlation ({value:.3f}) between '{col1}' and '{col2}'"
                        )

        return ValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
            row_count=len(features),
            null_percentages=self._null_percentages(features),
        )
