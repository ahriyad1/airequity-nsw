"""
Produce 24-hour threshold-crossing forecasts for every operational station,
with and without each station's own sensor.

WHAT THIS DOES
    Fits two classifiers on the historical feature matrix — one using each
    station's own sensor history (monitored) and one without it
    (unmonitored) — then forecasts the next 24 hours for every station
    under both. Writes a small table the dashboard reads.

WHY TWO FORECASTS
    These stations have sensors, so the monitored forecast is the better
    one to act on. The unmonitored forecast is what the system would say at
    the same location if the sensor did not exist. Showing both side by side
    demonstrates the project's central finding directly: the penalty for
    having no local sensor, measured in the leave-one-station-out study as a
    14.3% loss of precision, visible for each station on the day.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.build_features import (  # noqa: E402
    add_temporal_features, handle_gaps, regularise_index)
from src.features.spatial_features import (  # noqa: E402
    bearing_matrix, build_spatial, haversine_matrix, load_coordinates)

try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except (ImportError, OSError):
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_LGBM = False

FEATURES_DIR = Path("data/processed/features_spatial")
OBS_DIR = Path("data/processed/observations")
FORECAST_WEATHER = Path("data/processed/weather_forecasts_latest.parquet")
OUT_PATH = Path("data/processed/latest_forecast.parquet")
META_PATH = Path("data/processed/latest_forecast_meta.json")

TIMEZONE = "Australia/Sydney"
STALE_WARNING_HOURS = 72
MIN_HISTORY_HOURS = 168

WEATHER_RENAME = {
    "temp_forecast_c": "TEMP",
    "humid_forecast_pct": "HUMID",
    "wsp_forecast_m_s": "WSP",
    "wdr_forecast_deg": "WDR",
}

SPATIAL_PREFIX = "spatial_"
OWN_SENSOR_PREFIXES = ("pm25_lag_", "pm25_mean_", "pm25_max_", "pm25_std_",
                       "pm25_delta_")
OWN_SENSOR_EXACT = {"PM2.5", "imputed"}
WEATHER = ["TEMP", "HUMID", "WSP", "WDR", "wind_u", "wind_v",
           "temp_lag_24h", "humid_lag_24h", "wsp_lag_24h", "wdr_lag_24h"]
CALENDAR = ["hour", "dayofweek", "month", "is_weekend",
            "hour_sin", "hour_cos", "month_sin", "month_cos"]
GEO = ["latitude", "longitude"]

# Cost-optimal thresholds from the pooled threshold sweep
DEFAULT_THRESHOLD_MONITORED = 0.065
DEFAULT_THRESHOLD_UNMONITORED = 0.090

# Measured penalty for no local sensor, from the bootstrap analysis
PRECISION_PENALTY = 0.143
PRECISION_CI = (0.050, 0.238)


def feature_sets(df):
    """Same split as the leave-one-station-out validation."""
    spatial = [c for c in df.columns if c.startswith(SPATIAL_PREFIX)]
    own = [c for c in df.columns
           if c.startswith(OWN_SENSOR_PREFIXES) or c in OWN_SENSOR_EXACT]
    weather = [c for c in WEATHER if c in df.columns]
    cal = [c for c in CALENDAR if c in df.columns]
    geo = [c for c in GEO if c in df.columns]
    unmonitored = spatial + weather + cal + geo
    monitored = unmonitored + own
    return monitored, unmonitored, own


def make_model():
    #Same configuration as the validated model.
    if HAVE_LGBM:
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=42)
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=50, random_state=42)


def load_forecast_weather():
    #Load forecast weather and align it to the training column names
    if not FORECAST_WEATHER.exists():
        raise FileNotFoundError(
            f"{FORECAST_WEATHER} not found — run fetch_weather_openMeteo.py first.")

    wx = pd.read_parquet(FORECAST_WEATHER).rename(columns=WEATHER_RENAME)
    wx["timestamp"] = pd.to_datetime(wx["forecast_timestamp_local"])
    wx["site_id"] = pd.to_numeric(wx["site_id"], errors="coerce")
    wx = wx.dropna(subset=["site_id"])
    wx["site_id"] = wx["site_id"].astype(int)

    keep = ["site_id", "site_name", "region", "latitude", "longitude",
            "timestamp", "TEMP", "HUMID", "WSP", "WDR"]
    wx = wx[[c for c in keep if c in wx.columns]]

    rad = np.deg2rad(wx["WDR"])
    wx["wind_u"] = -wx["WSP"] * np.sin(rad)
    wx["wind_v"] = -wx["WSP"] * np.cos(rad)

    wx = wx.sort_values(["site_id", "timestamp"])
    for col in ["TEMP", "HUMID", "WSP", "WDR"]:
        wx[f"{col.lower()}_lag_24h"] = wx.groupby("site_id")[col].shift(24)
    return wx


def add_calendar(df):
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def recent_observations(operational_ids, lookback_hours):
   #Recent hourly observations, pivoted to one column per parameter
    if not OBS_DIR.exists():
        raise FileNotFoundError(f"{OBS_DIR} not found — run write_parquet.py first.")

    obs = pd.read_parquet(OBS_DIR)
    obs = obs[(obs["frequency"] == "Hourly average")
              & obs["parameter"].isin(["PM2.5", "TEMP", "HUMID", "WSP", "WDR"])
              & obs["site_id"].isin(operational_ids)]

    latest_ts = obs["timestamp"].max()
    obs = obs[obs["timestamp"] > latest_ts - pd.Timedelta(hours=lookback_hours)]

    wide = obs.pivot_table(index=["site_id", "site_name", "region", "timestamp"],
                           columns="parameter", values="value",
                           aggfunc="first").reset_index()
    wide.columns.name = None
    return wide, latest_ts


def latest_spatial_state(wide):
    #Spatial features from the latest network state, one row per station
    df = wide[["site_id", "timestamp", "PM2.5", "WDR"]].copy()
    df["PM2.5"] = df["PM2.5"].clip(lower=0)
    recent = df[df["timestamp"] > df["timestamp"].max() - pd.Timedelta(hours=48)]

    coords = load_coordinates()
    coords = coords[coords["site_id"].isin(recent["site_id"].unique())]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sp = build_spatial(recent, haversine_matrix(coords),
                           bearing_matrix(coords), k=3, idw_power=2.0)

    cols = [c for c in sp.columns if c.startswith(SPATIAL_PREFIX)]
    return (sp.dropna(subset=["spatial_idw"])
              .sort_values("timestamp")
              .groupby("site_id")[cols].last()
              .reset_index())


def latest_own_state(wide, own_cols):
    #Each station's own-sensor features at its latest observed hour
    df = regularise_index(wide.copy())
    df = handle_gaps(df)
    df = add_temporal_features(df)

    present = [c for c in own_cols if c in df.columns]
    latest = (df.dropna(subset=["PM2.5"])
                .sort_values("timestamp")
                .groupby("site_id")
                .tail(1)[["site_id", "timestamp"] + present]
                .rename(columns={"timestamp": "own_obs_time"}))
    return latest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold-monitored", type=float,
                    default=DEFAULT_THRESHOLD_MONITORED)
    ap.add_argument("--threshold-unmonitored", type=float,
                    default=DEFAULT_THRESHOLD_UNMONITORED)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--pm25-threshold", type=float, default=25.0)
    ap.add_argument("--lookback-hours", type=int, default=240,
                    help="Hours of recent observations used to build features "
                         "(default 240; must exceed 168 for the longest lag)")
    args = ap.parse_args()

    now = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None).floor("h")
    print(f"Issue time: {now:%Y-%m-%d %H:%M} ({TIMEZONE})\n")

    # ---- training ------------------------------------------------------
    print("Loading historical features...")
    hist = pd.read_parquet(FEATURES_DIR).dropna(subset=["label"])
    hist["label"] = hist["label"].astype(int)
    mon_feats, unmon_feats, own_cols = feature_sets(hist)
    operational = set(hist["site_id"].unique())
    print(f"  {len(hist):,} rows, {len(operational)} stations")
    print(f"  monitored: {len(mon_feats)} features, "
          f"unmonitored: {len(unmon_feats)} features")

    print("Fitting models...")
    y = hist["label"].values
    model_mon = make_model().fit(hist[mon_feats], y)
    model_unmon = make_model().fit(hist[unmon_feats], y)
    print(f"  both fitted "
          f"({'LightGBM' if HAVE_LGBM else 'HistGradientBoosting'})")

    # ---- inputs ----
    print("Loading forecast weather...")
    wx = load_forecast_weather()
    horizon_end = now + pd.Timedelta(hours=args.horizon)
    wx = wx[(wx["timestamp"] > now) & (wx["timestamp"] <= horizon_end)
            & wx["site_id"].isin(operational)]
    print(f"  {len(wx):,} station-hours in the next {args.horizon}h")
    if wx.empty:
        print("\nNo forecast hours in the window. Re-run fetch_weather_openMeteo.py.")
        return

    print("Loading recent observations...")
    wide, obs_time = recent_observations(operational, args.lookback_hours)
    history_h = (obs_time - wide["timestamp"].min()).total_seconds() / 3600
    age_h = (now - obs_time).total_seconds() / 3600
    print(f"  {history_h:.0f}h of history, latest {obs_time:%Y-%m-%d %H:%M} "
          f"({age_h:.0f}h before issue)")
    if age_h > STALE_WARNING_HOURS:
        print(f"  WARNING: observations more than {STALE_WARNING_HOURS}h old.")
    if history_h < MIN_HISTORY_HOURS:
        print(f"  WARNING: under {MIN_HISTORY_HOURS}h of history — long lags "
              f"will be missing and the monitored forecast is degraded. Fetch "
              f"more observations.")

    print("Building spatial and own-sensor features...")
    spatial = latest_spatial_state(wide)
    own = latest_own_state(wide, own_cols)
    print(f"  spatial: {spatial['site_id'].nunique()} stations, "
          f"own-sensor: {own['site_id'].nunique()} stations")

    # ---- predict -------------------------------------------------------
    X = wx.merge(spatial, on="site_id", how="left")
    X = X.merge(own, on="site_id", how="left")
    X = add_calendar(X)
    for c in set(mon_feats) - set(X.columns):
        X[c] = np.nan

    print("Predicting...")
    X["prob_monitored"] = model_mon.predict_proba(X[mon_feats])[:, 1]
    X["prob_unmonitored"] = model_unmon.predict_proba(X[unmon_feats])[:, 1]
    X["alert_monitored"] = (X["prob_monitored"]
                            >= args.threshold_monitored).astype(int)
    X["alert_unmonitored"] = (X["prob_unmonitored"]
                              >= args.threshold_unmonitored).astype(int)

    out = X[["site_id", "site_name", "region", "latitude", "longitude",
             "timestamp", "prob_monitored", "prob_unmonitored",
             "alert_monitored", "alert_unmonitored",
             "TEMP", "HUMID", "WSP", "WDR"]].copy()
    out["issued_at"] = now
    out["observations_up_to"] = obs_time
    out["threshold_monitored"] = args.threshold_monitored
    out["threshold_unmonitored"] = args.threshold_unmonitored
    out["pm25_threshold"] = args.pm25_threshold

    recent = wide[["site_id", "site_name", "region", "timestamp", "PM2.5"]].copy()
    recent["PM2.5"] = recent["PM2.5"].clip(lower=0)
    recent.to_parquet(Path("data/processed/latest_observations.parquet"), index=False)
    
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(OUT_PATH)

    summary = (out.groupby(["site_id", "site_name", "region"])
                  .agg(peak_monitored=("prob_monitored", "max"),
                       peak_unmonitored=("prob_unmonitored", "max"),
                       hours_alert_monitored=("alert_monitored", "sum"),
                       hours_alert_unmonitored=("alert_unmonitored", "sum"))
                  .reset_index()
                  .sort_values("peak_monitored", ascending=False))

    with open(META_PATH, "w") as f:
        json.dump({
            "issued_at": now.isoformat(),
            "timezone": TIMEZONE,
            "observations_up_to": obs_time.isoformat(),
            "observation_age_hours": round(age_h, 1),
            "history_hours": round(history_h, 1),
            "horizon_hours": args.horizon,
            "pm25_threshold_ug_m3": args.pm25_threshold,
            "threshold_monitored": args.threshold_monitored,
            "threshold_unmonitored": args.threshold_unmonitored,
            "model": "LightGBM" if HAVE_LGBM else "HistGradientBoosting",
            "n_features_monitored": len(mon_feats),
            "n_features_unmonitored": len(unmon_feats),
            "n_stations": int(out["site_id"].nunique()),
            "hours_alert_monitored": int(out["alert_monitored"].sum()),
            "hours_alert_unmonitored": int(out["alert_unmonitored"].sum()),
            "precision_penalty_unmonitored": PRECISION_PENALTY,
            "precision_penalty_ci": list(PRECISION_CI),
            "generated": datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2)

    print(f"\nWritten to {OUT_PATH}")
    print(f"  {len(out):,} station-hours, {out['site_id'].nunique()} stations")
    print(f"  hours on alert — with sensor: {int(out['alert_monitored'].sum())}, "
          f"without sensor: {int(out['alert_unmonitored'].sum())}")

    print("\nPeak probability by station (with sensor / without sensor)")
    print(summary.head(10)[["site_name", "region", "peak_monitored",
                            "peak_unmonitored", "hours_alert_monitored",
                            "hours_alert_unmonitored"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()