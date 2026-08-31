import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OBS_DIR = Path("data/processed/observations")
OUT_DIR = Path("data/processed/features")

TARGET = "PM2.5"
WEATHER = ["TEMP", "HUMID", "WSP", "WDR"]
LAGS = [1, 2, 3, 6, 12, 24, 48, 168]      # hours
ROLLING = [3, 6, 24, 72]                   # hours
MAX_GAP_HOURS = 3                          # interpolate gaps up to this long


def load_hourly():
    """Load hourly-average records and pivot parameters into columns."""
    df = pd.read_parquet(OBS_DIR)
    df = df[df["frequency"] == "Hourly average"]
    df = df[df["parameter"].isin([TARGET] + WEATHER)]

    wide = df.pivot_table(
        index=["site_id", "site_name", "region", "timestamp"],
        columns="parameter",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide.sort_values(["site_id", "timestamp"])


def regularise_index(df):
    """Give every station a continuous hourly index."""
    frames = []
    for sid, g in df.groupby("site_id", sort=False):
        g = g.set_index("timestamp").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(full)
        g["site_id"] = sid
        g["site_name"] = g["site_name"].ffill().bfill()
        g["region"] = g["region"].ffill().bfill()
        g.index.name = "timestamp"
        frames.append(g.reset_index())
    return pd.concat(frames, ignore_index=True)


def handle_gaps(df):
    """Interpolate short gaps only."""
    cols = [TARGET] + [c for c in WEATHER if c in df.columns]
    out = []
    for _, g in df.groupby("site_id", sort=False):
        g = g.sort_values("timestamp").copy()
        g["imputed"] = g[TARGET].isna()
        for c in cols:
            if c in g.columns:
                g[c] = g[c].interpolate(method="linear", limit=MAX_GAP_HOURS,
                                        limit_area="inside")
        g["imputed"] = g["imputed"] & g[TARGET].notna()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def add_temporal_features(df):
    """Lags and rolling statistics, computed per station, backwards only."""
    out = []
    for _, g in df.groupby("site_id", sort=False):
        g = g.sort_values("timestamp").copy()

        for lag in LAGS:
            g[f"pm25_lag_{lag}h"] = g[TARGET].shift(lag)

        for w in ROLLING:
            # closed='left' excludes the current observation, so the
            # rolling window describes the past, not the present
            r = g[TARGET].shift(1).rolling(w, min_periods=max(2, w // 4))
            g[f"pm25_mean_{w}h"] = r.mean()
            g[f"pm25_max_{w}h"] = r.max()
            g[f"pm25_std_{w}h"] = r.std()

        # rate of change
        g["pm25_delta_1h"] = g[TARGET] - g[f"pm25_lag_1h"]
        g["pm25_delta_24h"] = g[TARGET] - g[f"pm25_lag_24h"]

        for c in WEATHER:
            if c in g.columns:
                g[f"{c.lower()}_lag_24h"] = g[c].shift(24)

        out.append(g)
    return pd.concat(out, ignore_index=True)


def add_calendar_features(df):
    """Cyclical encodings so the model sees 23:00 and 00:00 as adjacent."""
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Wind direction is circular: 359 degrees and 1 degree are adjacent.
    # Decomposing into u/v components preserves that.
    if "WDR" in df.columns and "WSP" in df.columns:
        rad = np.deg2rad(df["WDR"])
        df["wind_u"] = -df["WSP"] * np.sin(rad)
        df["wind_v"] = -df["WSP"] * np.cos(rad)
    return df


def add_label(df, horizon, threshold, secondary):
    """Label: does PM2.5 exceed the threshold `horizon` hours from now?"""
    out = []
    for _, g in df.groupby("site_id", sort=False):
        g = g.sort_values("timestamp").copy()
        future = g[TARGET].shift(-horizon)
        g["target_pm25_future"] = future
        g["label"] = (future > threshold).astype("Int64")
        g.loc[future.isna(), "label"] = pd.NA
        g["label_severe"] = (future > secondary).astype("Int64")
        g.loc[future.isna(), "label_severe"] = pd.NA
        out.append(g)
    return pd.concat(out, ignore_index=True)


def verify_no_leakage(df, horizon):
    """
    Sanity check: the strongest feature must not be near-perfectly
    correlated with the label. If it is, something is leaking.
    """
    feats = [c for c in df.columns if c.startswith("pm25_lag_")
             or c.startswith("pm25_mean_")]
    sub = df.dropna(subset=["label"])
    if sub.empty:
        return
    corrs = {c: abs(sub[c].corr(sub["label"].astype(float)))
             for c in feats if sub[c].notna().any()}
    worst = max(corrs.items(), key=lambda kv: kv[1])
    print(f"\nLeakage check (horizon {horizon}h)")
    print(f"  strongest feature-label correlation: "
          f"{worst[0]} = {worst[1]:.3f}")
    if worst[1] > 0.95:
        print("  WARNING: correlation above 0.95 suggests leakage")
    else:
        print("  OK - no feature is near-perfectly predictive")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=24,
                    help="Forecast horizon in hours (default 24)")
    ap.add_argument("--threshold", type=float, default=25.0,
                    help="Primary threshold, ug/m3 (default 25)")
    ap.add_argument("--secondary", type=float, default=62.1,
                    help="Severe threshold, ug/m3 (default 62.1)")
    args = ap.parse_args()

    print("Loading observations...")
    df = load_hourly()
    print(f"  {len(df):,} station-hours, {df['site_id'].nunique()} stations")

    print("Regularising hourly index...")
    df = regularise_index(df)
    print(f"  {len(df):,} rows after filling implicit gaps")

    print(f"Interpolating gaps up to {MAX_GAP_HOURS}h...")
    df = handle_gaps(df)
    imputed = int(df["imputed"].sum())
    print(f"  {imputed:,} values imputed "
          f"({imputed / len(df):.1%} of rows)")

    print("Building temporal features...")
    df = add_temporal_features(df)

    print("Building calendar and wind features...")
    df = add_calendar_features(df)

    print(f"Labelling: PM2.5 > {args.threshold} at t+{args.horizon}h...")
    df = add_label(df, args.horizon, args.threshold, args.secondary)

    # Rows without a full lag history cannot be used
    before = len(df)
    df = df.dropna(subset=["label", f"pm25_lag_{max(LAGS)}h"])
    print(f"  {before - len(df):,} rows dropped "
          f"(incomplete history or no future value)")

    verify_no_leakage(df, args.horizon)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df["year"] = df["timestamp"].dt.year
    df.to_parquet(OUT_DIR, partition_cols=["year"], index=False)

    pos = int(df["label"].sum())
    pos_sev = int(df["label_severe"].sum())
    print(f"\nWritten to {OUT_DIR}/")
    print(f"  rows            : {len(df):,}")
    print(f"  features        : {len([c for c in df.columns if c.startswith(('pm25_', 'hour_', 'month_', 'wind_', 'temp_', 'humid_', 'wsp_', 'wdr_'))])}")
    print(f"  stations        : {df['site_id'].nunique()}")
    print(f"  date range      : {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  positive (>{args.threshold})   : {pos:,} ({pos/len(df):.2%})")
    print(f"  positive (>{args.secondary}) : {pos_sev:,} ({pos_sev/len(df):.3%})")
    print(f"\n  A model predicting 'never' scores "
          f"{1 - pos/len(df):.2%} accuracy and warns nobody.")


if __name__ == "__main__":
    main()