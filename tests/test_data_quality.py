"""Data quality checks for the AirEquity pipeline."""

import pandas as pd
import pytest

FEATURES = "data/processed/observations"


def test_pm25_values_are_plausible():
    """PM2.5 cannot be negative and 2000 ug/m3 would be off the scale."""
    df = pd.read_parquet(FEATURES)
    pm = df[df["parameter"] == "PM2.5"]["value"].dropna()
    assert (pm >= 0).all(), "negative PM2.5 values found"
    assert (pm < 2000).all(), "implausibly high PM2.5 values found"


def test_required_columns_exist():
    df = pd.read_parquet(FEATURES)
    for col in ["site_id", "parameter", "timestamp", "value"]:
        assert col in df.columns, f"missing column: {col}"


def test_no_duplicate_readings():
    """One reading per station, parameter, frequency and hour."""
    df = pd.read_parquet(FEATURES)
    dupes = df.duplicated(
        subset=["site_id", "parameter", "frequency", "timestamp"]
    ).sum()
    assert dupes == 0, f"{dupes} duplicate readings"


def test_timestamps_are_hourly():
    df = pd.read_parquet(FEATURES)
    assert (df["timestamp"].dt.minute == 0).all(), "non-hourly timestamps"
    