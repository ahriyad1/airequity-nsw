"""
Convert raw NSW Air Quality API responses into partitioned Parquet.

Write Parquet output and verify DuckDB query
"""

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed/observations")
SITES_FILE = Path("data/raw/sites.json")


def load_raw_file(path):
    """Flatten one raw JSON file into a DataFrame."""
    with open(path) as f:
        records = json.load(f)
    if not records:
        return pd.DataFrame()

    rows = []
    for r in records:
        p = r.get("Parameter") or {}
        rows.append({
            "site_id": r.get("Site_Id"),
            "parameter": p.get("ParameterCode"),
            "parameter_desc": p.get("ParameterDescription"),
            "units": p.get("Units"),
            "category": p.get("Category"),
            "subcategory": p.get("SubCategory"),
            "frequency": p.get("Frequency"),
            "date": r.get("Date"),
            "hour": r.get("Hour"),
            "value": r.get("Value"),
            "aqi_category": r.get("AirQualityCategory"),
            "determining_pollutant": r.get("DeterminingPollutant"),
        })
    return pd.DataFrame(rows)


def build_timestamp(df):
    # Combine Date and Hour into a proper timestamp.
    df["hour_start"] = df["hour"].astype("Int64") - 1
    df["timestamp"] = (
        pd.to_datetime(df["date"], errors="coerce")
        + pd.to_timedelta(df["hour_start"], unit="h")
    )
    return df


def attach_site_metadata(df):
    #Add station name and region if a cached site list is available
    if not SITES_FILE.exists():
        return df
    with open(SITES_FILE) as f:
        sites = json.load(f)
    meta = pd.DataFrame([{
        "site_id": s.get("Site_Id"),
        "site_name": s.get("SiteName"),
        "region": s.get("Region"),
        "latitude": s.get("Latitude"),
        "longitude": s.get("Longitude"),
    } for s in sites])
    return df.merge(meta, on="site_id", how="left")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frequency", default=None,
                    help='Keep only one frequency, e.g. "Hourly average". '
                         'Default keeps all variants.')
    ap.add_argument("--stats", action="store_true",
                    help="Print a summary of the processed data and exit")
    args = ap.parse_args()

    if args.stats:
        if not OUT_DIR.exists():
            print("No processed data yet. Run without --stats first.")
            return
        df = pd.read_parquet(OUT_DIR)
        print(f"Rows            : {len(df):,}")
        print(f"Stations        : {df['site_id'].nunique()}")
        print(f"Parameters      : {sorted(df['parameter'].dropna().unique())}")
        print(f"Date range      : {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Missing values  : {df['value'].isna().sum():,} "
              f"({df['value'].isna().mean():.1%})")
        return

    files = sorted(RAW_DIR.glob("obs_*.json"))
    if not files:
        print(f"No raw files found in {RAW_DIR}/")
        return

    print(f"Reading {len(files)} raw files...")
    frames = []
    for i, path in enumerate(files, 1):
        frames.append(load_raw_file(path))
        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", flush=True)

    df = pd.concat(frames, ignore_index=True)
    raw_rows = len(df)
    print(f"\nRaw rows              : {raw_rows:,}")

    df = df.drop_duplicates(
        subset=["site_id", "parameter", "frequency", "date", "hour"]
    )
    print(f"After deduplication   : {len(df):,} "
          f"({raw_rows - len(df):,} removed)")

    if args.frequency:
        before = len(df)
        df = df[df["frequency"] == args.frequency]
        print(f"After frequency filter: {len(df):,} "
              f"({before - len(df):,} removed)")

    df = build_timestamp(df)
    df = df.dropna(subset=["timestamp"])
    df = attach_site_metadata(df)

    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month

    import shutil
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR, partition_cols=["year", "month"], index=False)

    raw_mb = sum(f.stat().st_size for f in files) / 1e6
    out_mb = sum(f.stat().st_size for f in OUT_DIR.rglob("*.parquet")) / 1e6

    print(f"\nWritten to {OUT_DIR}/ (partitioned by year and month)")
    print(f"Raw JSON     : {raw_mb:,.0f} MB")
    print(f"Parquet      : {out_mb:,.0f} MB  "
          f"({raw_mb / out_mb:.0f}x smaller)")
    print(f"\nRows         : {len(df):,}")
    print(f"Stations     : {df['site_id'].nunique()}")
    print(f"Date range   : {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Missing      : {df['value'].isna().sum():,} "
          f"({df['value'].isna().mean():.1%})")


if __name__ == "__main__":
    main()
