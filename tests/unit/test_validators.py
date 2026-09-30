"""Unit tests for data validation."""

import numpy as np
import pandas as pd
import pytest

from src.data.validators import (
    DataValidator,
    SalesSchema,
    ValidationResult,
)


class TestSalesSchema:
    """Tests for sales data schema validation."""

    def test_valid_sales_data(self):
        """Valid sales data should pass validation."""
        df = pd.DataFrame(
            {
                "id": ["ITEM_1_STORE_1"] * 10,
                "item_id": ["ITEM_1"] * 10,
                "store_id": ["STORE_1"] * 10,
                "date": pd.date_range("2020-01-01", periods=10, freq="D"),
                "sales": np.random.exponential(10, 10),
            }
        )

        # Should not raise
        SalesSchema.validate(df)

    def test_missing_required_column(self):
        """Missing columns should fail validation."""
        df = pd.DataFrame(
            {
                "id": ["ITEM_1"] * 10,
                # Missing other required columns
            }
        )

        with pytest.raises(Exception):  # SchemaError
            SalesSchema.validate(df)

    def test_negative_sales(self):
        """Negative sales should fail validation."""
        df = pd.DataFrame(
            {
                "id": ["ITEM_1_STORE_1"] * 10,
                "item_id": ["ITEM_1"] * 10,
                "store_id": ["STORE_1"] * 10,
                "date": pd.date_range("2020-01-01", periods=10, freq="D"),
                "sales": [-5] * 10,  # Invalid negative values
            }
        )

        with pytest.raises(Exception):
            SalesSchema.validate(df)


class TestDataValidator:
    """Tests for DataValidator class."""

    @pytest.fixture
    def validator(self):
        return DataValidator()

    @pytest.fixture
    def valid_sales_df(self):
        return pd.DataFrame(
            {
                "id": ["ITEM_1_STORE_1"] * 10,
                "item_id": ["ITEM_1"] * 10,
                "store_id": ["STORE_1"] * 10,
                "date": pd.date_range("2020-01-01", periods=10, freq="D"),
                "sales": np.random.exponential(10, 10),
            }
        )

    def test_validate_returns_result(self, validator, valid_sales_df):
        """Validate should return ValidationResult."""
        result = validator.validate(valid_sales_df, "sales")

        assert isinstance(result, ValidationResult)

    def test_valid_data_passes(self, validator, valid_sales_df):
        """Valid data should pass validation."""
        result = validator.validate(valid_sales_df, "sales")

        assert result.is_valid
        assert len(result.errors) == 0

    def test_invalid_data_reports_errors(self, validator):
        """Invalid data should report errors."""
        df = pd.DataFrame(
            {
                "id": ["ITEM_1"] * 10,
                "sales": [-5] * 10,  # Missing columns and invalid values
            }
        )

        result = validator.validate(df, "sales")

        assert not result.is_valid
        assert len(result.errors) > 0

    def test_validate_all_datasets(
        self, validator, sample_sales_data, sample_calendar_data, sample_prices_data
    ):
        """Should validate all dataset types."""
        sales_result = validator.validate(sample_sales_data.to_pandas(), "sales")
        calendar_result = validator.validate(sample_calendar_data.to_pandas(), "calendar")
        prices_result = validator.validate(sample_prices_data.to_pandas(), "prices")

        # All should return results (may or may not be valid depending on fixtures)
        assert isinstance(sales_result, ValidationResult)
        assert isinstance(calendar_result, ValidationResult)
        assert isinstance(prices_result, ValidationResult)


class TestValidationResult:
    """Tests for ValidationResult model."""

    def test_is_valid_when_no_errors(self):
        """is_valid should be True when no errors."""
        result = ValidationResult(
            is_valid=True,
            n_rows=100,
            n_columns=10,
            errors=[],
            warnings=[],
        )

        assert result.is_valid

    def test_not_valid_when_errors(self):
        """is_valid should be False when errors present."""
        result = ValidationResult(
            is_valid=False,
            n_rows=100,
            n_columns=10,
            errors=["Column 'id' is missing"],
            warnings=[],
        )

        assert not result.is_valid
        assert len(result.errors) == 1
