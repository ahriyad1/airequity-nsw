"""
Build the processed observation dataset for AirEquity.

Reads raw NSW Air Quality API JSON files from data/raw/
and writes a Parquet dataset to data/processed/observations.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT = PROCESSED_DIR / "observations"


def load_raw_records() -> list[dict]:
    """Load all observation records from data/raw."""

    files = sorted(RAW_DIR.glob("obs_*.json"))

    if not files:
        raise FileNotFoundError(
            f"No observation files found in {RAW_DIR}"
        )

    records: list[dict] = []

    for path in files:
        print(f"Reading {path.name}...")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                f"{path.name} does not contain a JSON list"
            )

        records.extend(data)

    print(f"Loaded {len(records):,} raw records")

    return records


def parse_timestamp(record: dict) -> pd.Timestamp:
    """
    Convert the API Date + Hour fields into a timestamp.

    The API uses Hour values 1-24.
    Hour 1 = 00:00
    Hour 2 = 01:00
    ...
    Hour 24 = 23:00
    """

    date_value = record.get("Date")
    hour_value = record.get("Hour")

    if date_value is None or hour_value is None:
        return pd.NaT

    try:
        date = datetime.strptime(
            str(date_value),
            "%Y-%m-%d"
        )

        hour = int(hour_value)

        if hour < 1 or hour > 24:
            return pd.NaT

        timestamp = date + timedelta(hours=hour - 1)

        return pd.Timestamp(timestamp)

    except (ValueError, TypeError):
        return pd.NaT


def build_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert raw API records into a clean dataframe."""

    rows = []

    for record in records:
        timestamp = parse_timestamp(record)

        row = {
            "site_id": record.get("Site_Id"),
            "site_name": record.get("Site_Name"),
            "parameter": record.get("Parameter"),
            "value": record.get("Value"),
            "unit": record.get("Unit"),
            "frequency": record.get("Frequency"),
            "timestamp": timestamp,
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No records were loaded.")

    # ---------------------------------------------------------
    # Clean identifiers
    # ---------------------------------------------------------

    df["site_id"] = df["site_id"].astype("string")
    df["site_name"] = df["site_name"].astype("string")
    df["parameter"] = df["parameter"].astype("string")
    df["unit"] = df["unit"].astype("string")
    df["frequency"] = df["frequency"].astype("string")

    # ---------------------------------------------------------
    # Convert measurement values to numeric
    # ---------------------------------------------------------

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # Validate timestamps
    # ---------------------------------------------------------

    bad_timestamps = df["timestamp"].isna().sum()

    if bad_timestamps:
        raise ValueError(
            f"{bad_timestamps:,} records have invalid timestamps"
        )

    # ---------------------------------------------------------
    # Make sure timestamps are hourly
    # ---------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Some timestamps could not be converted to datetime."
        )

    # ---------------------------------------------------------
    # Remove exact duplicate observations
    #
    # IMPORTANT:
    # Only use simple scalar/hashable columns here.
    # Do NOT use the original API record dictionaries.
    # ---------------------------------------------------------

    duplicate_columns = [
        "site_id",
        "parameter",
        "frequency",
        "timestamp",
    ]

    print(
        f"Before duplicate removal: {len(df):,} records"
    )

    duplicates = df.duplicated(
        subset=duplicate_columns
    ).sum()

    if duplicates:
        print(
            f"Removing {duplicates:,} duplicate observations..."
        )

        df = df.drop_duplicates(
            subset=duplicate_columns,
            keep="first"
        )

    print(
        f"After duplicate removal: {len(df):,} records"
    )

    # ---------------------------------------------------------
    # Sort the final dataset
    # ---------------------------------------------------------

    df = df.sort_values(
        [
            "site_id",
            "parameter",
            "timestamp",
        ]
    ).reset_index(drop=True)

    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    """Run basic validation before writing the dataset."""

    required_columns = [
        "site_id",
        "parameter",
        "timestamp",
        "value",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Timestamp validation
    if df["timestamp"].isna().any():
        raise ValueError(
            "Dataset contains invalid timestamps."
        )

    # Check timestamps are exactly on the hour
    non_hourly = (
        (df["timestamp"].dt.minute != 0)
        | (df["timestamp"].dt.second != 0)
        | (df["timestamp"].dt.microsecond != 0)
    ).sum()

    if non_hourly:
        raise ValueError(
            f"{non_hourly:,} records have non-hourly timestamps."
        )

    # PM2.5 sanity check
    pm25 = df.loc[
        df["parameter"].str.upper() == "PM2.5",
        "value",
    ].dropna()

    if not pm25.empty:

        negative = (pm25 < 0).sum()

        if negative:
            raise ValueError(
                f"{negative:,} negative PM2.5 values found."
            )

        extremely_high = (pm25 >= 2000).sum()

        if extremely_high:
            raise ValueError(
                f"{extremely_high:,} implausibly high PM2.5 values found."
            )

    # Duplicate validation
    duplicate_columns = [
        "site_id",
        "parameter",
        "frequency",
        "timestamp",
    ]

    duplicates = df.duplicated(
        subset=duplicate_columns
    ).sum()

    if duplicates:
        raise ValueError(
            f"{duplicates:,} duplicate readings remain."
        )

    print("Validation passed.")


def save_dataframe(df: pd.DataFrame) -> None:
    """Write the processed dataframe as Parquet."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Pandas/pyarrow can write a parquet dataset
    # without requiring a .parquet extension.
    df.to_parquet(
        OUTPUT,
        index=False
    )

    print(f"Saved to: {OUTPUT}")


def main() -> None:
    """Run the complete feature-building pipeline."""

    records = load_raw_records()

    df = build_dataframe(records)

    print("\nFinal columns:")
    print(list(df.columns))

    print("\nFinal shape:")
    print(f"{len(df):,} rows x {len(df.columns)} columns")

    print("\nValidating dataset...")

    validate_dataframe(df)

    save_dataframe(df)


if __name__ == "__main__":
    main()