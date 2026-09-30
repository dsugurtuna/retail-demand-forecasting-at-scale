"""Unit tests for data validation."""

import numpy as np
import pandas as pd
import pytest

from src.data.validators import (
    CalendarDataSchema,
    DataValidator,
    PriceDataSchema,
    SalesDataSchema,
    ValidationResult,
)


def _sales(days: int = 60, sales=None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["ITEM_1_STORE_1"] * days,
            "item_id": ["ITEM_1"] * days,
            "store_id": ["STORE_1"] * days,
            "date": pd.date_range("2020-01-01", periods=days, freq="D"),
            "sales": np.random.default_rng(0).exponential(10, days) if sales is None else sales,
        }
    )


class TestSalesSchema:
    def test_valid_sales_data(self):
        SalesDataSchema.validate(_sales())  # should not raise

    def test_missing_required_column(self):
        with pytest.raises(Exception, match="column"):
            SalesDataSchema.validate(pd.DataFrame({"id": ["ITEM_1"] * 10}))

    def test_negative_sales(self):
        with pytest.raises(Exception, match="greater_than_or_equal_to"):
            SalesDataSchema.validate(_sales(10, sales=[-5] * 10))

    def test_calendar_and_price_schemas_exist(self):
        assert "wday" in CalendarDataSchema.to_schema().columns
        assert "sell_price" in PriceDataSchema.to_schema().columns


class TestDataValidator:
    @pytest.fixture
    def validator(self):
        return DataValidator()

    def test_validate_returns_result(self, validator):
        assert isinstance(validator.validate(_sales(), "sales"), ValidationResult)

    def test_valid_data_passes(self, validator):
        result = validator.validate(_sales(), "sales")
        assert result.is_valid
        assert result.errors == []

    def test_invalid_data_reports_errors(self, validator):
        df = pd.DataFrame({"id": ["ITEM_1"] * 10, "sales": [-5] * 10})
        result = validator.validate(df, "sales")
        assert not result.is_valid
        assert "Missing required column 'date'" in result.errors
        assert any("negative" in e for e in result.errors)

    def test_short_history_is_an_error(self, validator):
        result = validator.validate(_sales(days=10), "sales")
        assert any("Insufficient history" in e for e in result.errors)

    def test_duplicates_are_an_error(self, validator):
        df = _sales()
        result = validator.validate(pd.concat([df, df.head(3)]), "sales")
        assert any("duplicate" in e for e in result.errors)

    def test_nulls_in_optional_columns_only_warn(self, validator, m5_data):
        result = validator.validate(m5_data, "sales")
        assert result.is_valid, result.errors
        assert any("event_name_1" in w for w in result.warnings)

    def test_raise_on_error(self, validator):
        with pytest.raises(ValueError, match="validation failed"):
            validator.validate(_sales(days=5), "sales", raise_on_error=True)

    def test_unknown_schema(self, validator):
        with pytest.raises(ValueError, match="Unknown schema"):
            validator.validate(_sales(), "nope")

    def test_validate_all_datasets(self, validator, m5_frames):
        _, calendar, prices = m5_frames
        assert validator.validate(calendar, "calendar").is_valid
        assert validator.validate(prices, "prices").is_valid

    def test_validate_features_flags_infinities(self, validator):
        features = pd.DataFrame({"a": [1.0, np.inf, 2.0], "b": [1.0, 1.0, 1.0]})
        result = validator.validate_features(features)
        assert not result.is_valid
        assert any("zero variance" in w for w in result.warnings)


class TestValidationResult:
    def test_is_valid_when_no_errors(self):
        result = ValidationResult(is_valid=True, row_count=100, errors=[], warnings=[])
        assert result.is_valid

    def test_not_valid_when_errors(self):
        result = ValidationResult(
            is_valid=False, row_count=100, errors=["Column 'id' is missing"], warnings=None
        )
        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.warnings == []
