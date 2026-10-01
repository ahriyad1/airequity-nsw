"""
Data quality checks for the AirEquity pipeline.

Two layers are tested separately because they have different contracts:

    data/processed/observations   raw readings as the API reported them.
                                  May contain small negative PM2.5 values
                                  from instrument baseline noise.

    data/processed/features       modelling table. Negatives have been
                                  clipped, gaps handled, labels attached.

Run:
    python3 -m pytest tests/ -v
"""

from pathlib import Path

import pandas as pd
import pytest

OBSERVATIONS = Path("data/processed/observations")
FEATURES = Path("data/processed/features")

# Optical PM2.5 instruments report small negatives in clean air. Measured
# floor across the 2023-2024 dataset is -10.00 ug/m3, affecting 7.6% of
# readings. Anything below this indicates a fault rather than noise.
NOISE_FLOOR = -10.0
IMPLAUSIBLE_HIGH = 2000.0


@pytest.fixture(scope="module")
def observations():
    if not OBSERVATIONS.exists():
        pytest.skip("observations not built — run write_parquet.py first")
    return pd.read_parquet(OBSERVATIONS)


@pytest.fixture(scope="module")
def features():
    if not FEATURES.exists():
        pytest.skip("features not built — run build_features.py first")
    return pd.read_parquet(FEATURES)


# ---------------------------------------------------------------- raw layer

def test_pm25_values_are_plausible(observations):
    """
    Raw PM2.5 may sit slightly below zero — optical instruments report
    baseline noise in clean air. Values below the measured noise floor, or
    implausibly high, indicate instrument or ingestion faults.

    Note: the 12 September 2023 event produced genuine readings above
    600 ug/m3 across multiple stations. The upper bound must not exclude
    real extreme events, which are the phenomenon this project forecasts.
    """
    pm = observations.loc[observations["parameter"] == "PM2.5", "value"].dropna()
    assert not pm.empty, "no PM2.5 readings found"

    below = pm[pm < NOISE_FLOOR]
    assert below.empty, (
        f"{len(below)} PM2.5 readings below the {NOISE_FLOOR} noise floor "
        f"(min {below.min():.2f}) — likely instrument fault"
    )
    above = pm[pm >= IMPLAUSIBLE_HIGH]
    assert above.empty, (
        f"{len(above)} PM2.5 readings at or above {IMPLAUSIBLE_HIGH} "
        f"(max {above.max():.2f}) — implausible"
    )


def test_required_columns_exist(observations):
    for col in ["site_id", "parameter", "timestamp", "value", "frequency"]:
        assert col in observations.columns, f"missing column: {col}"


def test_no_duplicate_readings(observations):
    """
    One reading per station, parameter, frequency and hour.

    This check exists because writing partitioned Parquet appends rather
    than overwrites: re-running the conversion without clearing the output
    directory silently doubles every row.
    """
    dupes = observations.duplicated(
        subset=["site_id", "parameter", "frequency", "timestamp"]
    ).sum()
    assert dupes == 0, f"{dupes} duplicate readings"


def test_timestamps_are_hourly(observations):
    ts = observations["timestamp"].dropna()
    assert (ts.dt.minute == 0).all(), "timestamps not aligned to the hour"
    assert (ts.dt.second == 0).all(), "timestamps carry seconds"


def test_hours_within_expected_range(observations):
    """
    The API reports Hour 1-24, where hour 1 means 12am-1am. build_timestamp()
    subtracts one to give a 0-23 clock hour. Values outside that range mean
    the conversion has drifted.
    """
    h = observations["timestamp"].dt.hour.dropna()
    assert h.between(0, 23).all(), "clock hours outside 0-23"


# ----------------------------------------------------------- feature layer

def test_features_have_no_negative_pm25(features):
    """
    Negatives are clipped to zero during feature engineering. They are
    clipped rather than dropped because they are valid clean-air
    observations; removing 7.6% of readings would bias the distribution
    upward for a threshold-crossing task.
    """
    pm = features["PM2.5"].dropna()
    assert (pm >= 0).all(), (
        f"{int((pm < 0).sum())} negative PM2.5 values survived clipping"
    )


def test_labels_are_binary(features):
    labels = features["label"].dropna().unique()
    assert set(labels) <= {0, 1}, f"unexpected label values: {sorted(labels)}"


def test_no_duplicate_station_hours(features):
    dupes = features.duplicated(subset=["site_id", "timestamp"]).sum()
    assert dupes == 0, f"{dupes} duplicate station-hours in features"


def test_lag_features_are_backward_looking(features):
    """
    Guard against leakage. A lag feature must correlate less strongly with
    the future label than a near-perfect predictor would. Correlation above
    0.95 would indicate a feature has been built from future data.
    """
    sub = features.dropna(subset=["label", "pm25_lag_1h"])
    if sub.empty:
        pytest.skip("no labelled rows")
    corr = abs(sub["pm25_lag_1h"].corr(sub["label"].astype(float)))
    assert corr < 0.95, (
        f"pm25_lag_1h correlates {corr:.3f} with the label — possible leakage"
    )