"""
Daily PM2.5 advisory forecast for every station, with and without each
station's own sensor.

WHAT IT PREDICTS
    The chance that a station's daily-average PM2.5 goes over 25 ug/m3
    (the national 24-hour standard), for today and for tomorrow.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.daily_target import (  # noqa: E402
    MIN_HOURS, TAU, THRESHOLD, build_table, daily_features, load_daily,
    make_model)

HISTORY_DIR = Path("data/processed/history/observations")
RECENT_DIR = Path("data/processed/observations")
RESULTS_DIR = Path("results")
OUT_PATH = Path("data/processed/latest_daily_forecast.parquet")
META_PATH = Path("data/processed/latest_daily_forecast_meta.json")
RECENT_PATH = Path("data/processed/latest_observations.parquet")

TIMEZONE = "Australia/Sydney"
MODEL = "logistic"
MAX_DELAY = 3            # beyond this the inputs are too old to forecast
MIN_STATION_SHARE = 0.5  # share of stations needed for a day to count as complete

# Validation results for each delay, produced by daily_target.py
VALIDATION_FILES = {
    0: {"2019-20 smoke": "daily_target_logistic_smoke_2019_20_summary.csv",
        "2021-23": "daily_target_logistic_2021_23_summary.csv",
        "2024-25": "daily_target_logistic_2024_25_summary.csv"},
    1: {"2019-20 smoke": "daily_target_logistic_delay1_smoke_2019_20_summary.csv",
        "2021-23": "daily_target_logistic_delay1_2021_23_summary.csv",
        "2024-25": "daily_target_logistic_delay1_2024_25_summary.csv"},
    2: {"2019-20 smoke": "daily_target_logistic_smoke_2019_20_summary.csv",
        "2021-23": "daily_target_logistic_2021_23_summary.csv",
        "2024-25": "daily_target_logistic_2024_25_summary.csv"},
}
METHODS = ["with_sensor", "without_sensor", "persistence", "network_persistence"]


def latest_complete_day(daily):
   #Newest day on which most stations have a valid daily mean
    n_stations = daily["site_id"].nunique()
    valid = daily.dropna(subset=["pm_mean"]).groupby("day")["site_id"].nunique()
    ok = valid[valid >= max(3, MIN_STATION_SHARE * n_stations)]
    if ok.empty:
        sys.exit("No complete day in the recent observations; cannot forecast.")
    return ok.index.max()


def validation_for(delay):
 #Recall, precision and cost per test period for this delay
    out = {}
    for period, fname in VALIDATION_FILES.get(delay, {}).items():
        path = RESULTS_DIR / fname
        if not path.exists():
            continue
        s = pd.read_csv(path)
        s = s[(s["test_years"] == "all") & (s["delay_days"] == delay)].set_index("method")
        out[period] = {m: {k: (None if pd.isna(s.loc[m, k]) else round(float(s.loc[m, k]), 4))
                           for k in ["recall", "precision", "cost", "events", "caught",
                                     "false_alarms"]}
                       for m in METHODS if m in s.index}
    return out


def recent_hourly_pm():
    """Recent hourly PM2.5 per station, for the dashboard's charts."""
    cols = ["site_id", "site_name", "region", "parameter", "frequency", "timestamp", "value"]
    df = pd.read_parquet(RECENT_DIR, columns=cols)
    df = df[(df["parameter"] == "PM2.5") & (df["frequency"] == "Hourly average")]
    df = df.drop_duplicates(["site_id", "timestamp"])
    out = df[["site_id", "site_name", "region", "timestamp"]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).astype("datetime64[ns]")
    out["PM2.5"] = pd.to_numeric(df["value"], errors="coerce").clip(lower=0)
    return out.sort_values(["site_id", "timestamp"]).reset_index(drop=True)


def atomic_write(df_or_text, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    if isinstance(df_or_text, pd.DataFrame):
        df_or_text.to_parquet(tmp, index=False)
    else:
        tmp.write_text(df_or_text)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--today", help="override today's date (YYYY-MM-DD), for testing")
    args = ap.parse_args()

    now = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None).floor("min")
    today = pd.Timestamp(args.today) if args.today else now.normalize()

    for p in (HISTORY_DIR, RECENT_DIR):
        if not p.exists():
            sys.exit(f"Missing {p}. See the workflow for how it is restored.")

    print("Loading recent observations...")
    recent_daily, recent_sites = load_daily(RECENT_DIR)
    base_day = latest_complete_day(recent_daily)
    hourly = recent_hourly_pm()
    obs_to = hourly["timestamp"].max()
    print(f"  {recent_daily['site_id'].nunique()} stations, readings up to {obs_to}, "
          f"newest complete day {base_day.date()}")

    feat, _, _ = daily_features(recent_daily, recent_sites)
    base = feat[feat["base_day"] == base_day].copy()
    region = hourly.drop_duplicates("site_id").set_index("site_id")["region"]

    targets = []
    for horizon, day in [("today", today), ("tomorrow", today + pd.Timedelta(days=1))]:
        delay = (day - base_day).days - 1
        if delay < 0:
            continue
        if delay > MAX_DELAY:
            print(f"  {horizon}: readings are {delay + 1} days old; skipping")
            continue
        targets.append((horizon, day, delay))
    if not targets:
        sys.exit("Readings are too old to forecast today or tomorrow.")

    print("Loading 2016-2025 history...")
    hist_daily, hist_sites = load_daily(HISTORY_DIR)
    train_years = [int(hist_daily["day"].dt.year.min()), int(hist_daily["day"].dt.year.max())]

    rows, meta_targets = [], []
    for horizon, day, delay in targets:
        table, fsets = build_table(hist_daily, hist_sites, delay)
        cols = fsets[MODEL]
        probs = {}
        for cond in ("with_sensor", "without_sensor"):
            model = make_model(MODEL).fit(table[cols[cond]], table["y"])
            probs[cond] = model.predict_proba(base[cols[cond]])[:, 1]
        print(f"  {horizon} ({day.date()}): trained on {len(table):,} station-days, "
              f"{int(table['y'].sum())} over 25; inputs {delay + 1} day(s) old")

        out = pd.DataFrame({
            "site_id": base["site_id"].values,
            "horizon": horizon,
            "target_day": day,
            "base_day": base_day,
            "delay_days": delay,
            "prob_with_sensor": probs["with_sensor"],
            "prob_without_sensor": probs["without_sensor"],
            "last_daily_mean": base["pm_mean"].values,
            "neighbour_mean": base["net_idw"].values,
        })
        # Without a reading on the base day there is no own-sensor forecast
        out.loc[out["last_daily_mean"].isna(), "prob_with_sensor"] = np.nan
        out["alert_with_sensor"] = (out["prob_with_sensor"] >= TAU).astype(int)
        out["alert_without_sensor"] = (out["prob_without_sensor"] >= TAU).astype(int)
        rows.append(out)
        meta_targets.append({
            "horizon": horizon, "target_day": day.date().isoformat(),
            "delay_days": delay, "inputs_age_days": delay + 1,
            "stations_on_advisory_with_sensor": int(out["alert_with_sensor"].sum()),
            "stations_on_advisory_without_sensor": int(out["alert_without_sensor"].sum()),
            "validation": validation_for(delay),
        })

    fc = pd.concat(rows, ignore_index=True)
    names = recent_sites[["site_name", "latitude", "longitude"]]
    fc = fc.join(names, on="site_id")
    fc["region"] = fc["site_id"].map(region)
    fc["issued_at"] = now
    fc = fc[["site_id", "site_name", "region", "latitude", "longitude", "horizon",
             "target_day", "base_day", "delay_days", "prob_with_sensor",
             "prob_without_sensor", "alert_with_sensor", "alert_without_sensor",
             "last_daily_mean", "neighbour_mean", "issued_at"]]

    meta = {
        "issued_at": now.isoformat(),
        "timezone": TIMEZONE,
        "observations_up_to": obs_to.isoformat(),
        "base_day": base_day.date().isoformat(),
        "pm25_threshold_ug_m3": THRESHOLD,
        "min_valid_hours": MIN_HOURS,
        "decision_threshold": round(TAU, 4),
        "cost_ratio": 10,
        "model": "Logistic regression on pollution features (own sensor and neighbours)",
        "training_years": train_years,
        "n_stations": int(fc["site_id"].nunique()),
        "forecasts": meta_targets,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(fc, OUT_PATH)
    atomic_write(json.dumps(meta, indent=2), META_PATH)
    atomic_write(hourly, RECENT_PATH)

    print()
    for t in meta_targets:
        print(f"  {t['horizon']:<9} {t['target_day']}: "
              f"{t['stations_on_advisory_with_sensor']} station(s) on advisory with own sensor, "
              f"{t['stations_on_advisory_without_sensor']} without")
    print(f"\nWritten {OUT_PATH}, {META_PATH.name}, {RECENT_PATH.name}")


if __name__ == "__main__":
    main()