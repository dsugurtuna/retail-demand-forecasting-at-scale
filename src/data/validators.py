"""
Data validation schemas using Pandera.

This module provides comprehensive data validation for the forecasting pipeline,
ensuring data quality and catching issues early in the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import pandera as pa
import polars as pl
from pandera.typing import Series
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class SalesDataSchema(pa.DataFrameModel):
    """
    Pandera schema for validating sales data.
    
    This schema enforces:
    - Required columns with correct dtypes
    - Value range constraints
    - Null value thresholds
    """
    
    id: Series[str] = pa.Field(nullable=False, description="Unique item-store identifier")
    item_id: Series[str] = pa.Field(nullable=False, description="Item identifier")
    store_id: Series[str] = pa.Field(nullable=False, description="Store identifier")
    d: Series[str] = pa.Field(nullable=False, description="Day identifier (d_1, d_2, ...)")
    sales: Series[float] = pa.Field(
        ge=0, 
        nullable=True,
        description="Sales quantity (non-negative)"
    )
    date: Series[pd.Timestamp] = pa.Field(nullable=False, description="Date of sale")
    
    class Config:
        """Schema configuration."""
        coerce = True
        strict = False  # Allow additional columns


class CalendarDataSchema(pa.DataFrameModel):
    """Schema for calendar/events data."""
    
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
    """Schema for pricing data."""
    
    store_id: Series[str] = pa.Field(nullable=False)
    item_id: Series[str] = pa.Field(nullable=False)
    wm_yr_wk: Series[int] = pa.Field(ge=0, nullable=False)
    sell_price: Series[float] = pa.Field(
        gt=0,
        nullable=True,
        description="Selling price (positive)"
    )
    
    class Config:
        coerce = True
        strict = False


class ValidationResult(BaseModel):
    """Result of data validation."""
    
    is_valid: bool = Field(..., description="Whether validation passed")
    errors: list[str] = Field(default_factory=list, description="List of validation errors")
    warnings: list[str] = Field(default_factory=list, description="List of warnings")
    row_count: int = Field(default=0, description="Number of rows validated")
    null_percentages: dict[str, float] = Field(
        default_factory=dict, 
        description="Null percentage per column"
    )
    
    @field_validator("errors", "warnings", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> list:
        """Ensure errors and warnings are lists."""
        if v is None:
            return []
        return list(v)


class DataValidator:
    """
    Comprehensive data validator for the forecasting pipeline.
    
    Provides:
    - Schema validation using Pandera
    - Statistical validation (distributions, outliers)
    - Business rule validation
    - Temporal consistency checks
    
    Example:
        >>> validator = DataValidator()
        >>> result = validator.validate(df)
        >>> if not result.is_valid:
        ...     print(result.errors)
    """
    
    def __init__(
        self,
        null_threshold: float = 0.05,
        outlier_std_threshold: float = 4.0,
        min_history_days: int = 28
    ) -> None:
        """
        Initialize validator.
        
        Args:
            null_threshold: Maximum allowed null percentage per column
            outlier_std_threshold: Number of std devs for outlier detection
            min_history_days: Minimum required history days
        """
        self.null_threshold = null_threshold
        self.outlier_std_threshold = outlier_std_threshold
        self.min_history_days = min_history_days
        
        self._schemas = {
            "sales": SalesDataSchema,
            "calendar": CalendarDataSchema,
            "prices": PriceDataSchema,
        }
    
    def validate(
        self,
        data: pl.DataFrame | pd.DataFrame,
        schema_name: str = "sales",
        raise_on_error: bool = False
    ) -> ValidationResult:
        """
        Validate data against schema and business rules.
        
        Args:
            data: Data to validate
            schema_name: Name of schema to use
            raise_on_error: Whether to raise exception on validation failure
            
        Returns:
            ValidationResult with validation status and details
        """
        errors: list[str] = []
        warnings: list[str] = []
        
        # Convert Polars to Pandas for Pandera
        if isinstance(data, pl.DataFrame):
            df = data.to_pandas()
        else:
            df = data
        
        # Schema validation
        if schema_name in self._schemas:
            schema_errors = self._validate_schema(df, schema_name)
            errors.extend(schema_errors)
        
        # Null validation
        null_percentages = self._calculate_null_percentages(df)
        for col, pct in null_percentages.items():
            if pct > self.null_threshold:
                errors.append(
                    f"Column '{col}' has {pct:.1%} null values, "
                    f"exceeding threshold of {self.null_threshold:.1%}"
                )
        
        # Statistical validation
        stat_warnings = self._validate_statistics(df)
        warnings.extend(stat_warnings)
        
        # Business rules validation
        business_errors = self._validate_business_rules(df)
        errors.extend(business_errors)
        
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            row_count=len(df),
            null_percentages=null_percentages
        )
        
        if not result.is_valid:
            logger.error(f"Validation failed with {len(errors)} errors")
            for error in errors:
                logger.error(f"  - {error}")
            
            if raise_on_error:
                raise ValueError(f"Data validation failed: {errors}")
        else:
            logger.info(f"Validation passed for {len(df):,} rows")
        
        if warnings:
            for warning in warnings:
                logger.warning(f"  - {warning}")
        
        return result
    
    def _validate_schema(self, df: pd.DataFrame, schema_name: str) -> list[str]:
        """Validate against Pandera schema."""
        errors = []
        schema = self._schemas[schema_name]
        
        try:
            schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as exc:
            for failure in exc.failure_cases.itertuples():
                errors.append(
                    f"Schema error in column '{failure.column}': {failure.check}"
                )
        except Exception as e:
            errors.append(f"Schema validation error: {str(e)}")
        
        return errors
    
    def _calculate_null_percentages(self, df: pd.DataFrame) -> dict[str, float]:
        """Calculate null percentage for each column."""
        return (df.isnull().sum() / len(df)).to_dict()
    
    def _validate_statistics(self, df: pd.DataFrame) -> list[str]:
        """Validate statistical properties."""
        warnings = []
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        
        for col in numeric_cols:
            if col == "sales" and col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                
                if std > 0:
                    outlier_mask = (df[col] - mean).abs() > (self.outlier_std_threshold * std)
                    outlier_pct = outlier_mask.mean()
                    
                    if outlier_pct > 0.01:  # More than 1% outliers
                        warnings.append(
                            f"Column '{col}' has {outlier_pct:.1%} potential outliers "
                            f"({self.outlier_std_threshold} std from mean)"
                        )
        
        return warnings
    
    def _validate_business_rules(self, df: pd.DataFrame) -> list[str]:
        """Validate business-specific rules."""
        errors = []
        
        # Check minimum history
        if "date" in df.columns:
            date_range = (df["date"].max() - df["date"].min()).days
            if date_range < self.min_history_days:
                errors.append(
                    f"Insufficient history: {date_range} days, "
                    f"minimum required: {self.min_history_days}"
                )
        
        # Check for duplicate entries
        if {"id", "d"}.issubset(df.columns):
            duplicates = df.duplicated(subset=["id", "d"]).sum()
            if duplicates > 0:
                errors.append(f"Found {duplicates} duplicate id-date combinations")
        
        # Check sales non-negativity
        if "sales" in df.columns:
            negative_sales = (df["sales"] < 0).sum()
            if negative_sales > 0:
                errors.append(f"Found {negative_sales} negative sales values")
        
        return errors
    
    def validate_features(
        self,
        features: pd.DataFrame,
        target: str = "sales"
    ) -> ValidationResult:
        """
        Validate feature matrix before training.
        
        Args:
            features: Feature DataFrame
            target: Target column name
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        # Check for infinite values
        numeric_cols = features.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            inf_count = features[col].isin([float("inf"), float("-inf")]).sum()
            if inf_count > 0:
                errors.append(f"Column '{col}' contains {inf_count} infinite values")
        
        # Check feature variance
        for col in numeric_cols:
            if col != target and features[col].std() == 0:
                warnings.append(f"Column '{col}' has zero variance")
        
        # Check for highly correlated features
        if len(numeric_cols) > 1:
            corr_matrix = features[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    if abs(corr_matrix.loc[col1, col2]) > 0.99:
                        warnings.append(
                            f"High correlation ({corr_matrix.loc[col1, col2]:.3f}) "
                            f"between '{col1}' and '{col2}'"
                        )
        
        null_percentages = self._calculate_null_percentages(features)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            row_count=len(features),
            null_percentages=null_percentages
        )
