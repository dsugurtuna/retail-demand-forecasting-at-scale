"""Retail demand forecasting: LightGBM on M5-style data with leakage-safe features.

The package root deliberately imports nothing. Importing a submodule (for
example ``src.evaluation.metrics``) should not drag in FastAPI, LightGBM or
Pandera, and a broken optional component must not break every import.
"""

__version__ = "2.1.0"
