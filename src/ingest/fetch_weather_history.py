"""
Fetch archived day-ahead weather forecasts for every Sydney station.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
SITES_FILE = Path("data/raw/sites.json")
OUT_PATH = Path("data/processed/weather_history.parquet")
REGIONS = {"Sydney East", "Sydney South-west", "Sydney North-west"}
VARIABLES = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
             "wind_direction_10m", "precipitation", "surface_pressure", "cloud_cover"]
LEAD = "previous_day1"
TIMEOUT = 90
RETRIES = 4


def load_sites():
    sites = json.loads(SITES_FILE.read_text())
    out = []
    for s in sites:
        name = str(s.get("SiteName") or "")
        if s.get("Region") in REGIONS and s.get("Latitude") and "test" not in name.lower():
            out.append({"site_id": int(s["Site_Id"]), "site_name": name,
                        "latitude": float(s["Latitude"]), "longitude": float(s["Longitude"])})
    return out


def fetch(session, site, start, end, model):
    params = {
        "latitude": f"{site['latitude']:.4f}", "longitude": f"{site['longitude']:.4f}",
        "hourly": ",".join(f"{v}_{LEAD}" for v in VARIABLES),
        "models": model, "start_date": start, "end_date": end,
        "timezone": "Australia/Sydney", "wind_speed_unit": "ms",
    }
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(URL, params=params, timeout=TIMEOUT)
            if r.status_code == 400:   # bad request: report the API's reason and stop
                raise SystemExit(f"API rejected request: {r.text[:400]}")
            r.raise_for_status()
            hourly = r.json()["hourly"]
            df = pd.DataFrame(hourly).rename(columns=lambda c: c.replace(f"_{LEAD}", ""))
            df["time"] = pd.to_datetime(df["time"])
            return df
        except (requests.RequestException, KeyError, ValueError) as e:
            if attempt == RETRIES:
                raise
            wait = 5 * 2 ** attempt
            print(f"    retry {attempt} after {type(e).__name__}; waiting {wait}s")
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--model", default="jma_gsm")
    args = ap.parse_args()

    sites = load_sites()
    years = range(int(args.start[:4]), int(args.end[:4]) + 1)
    print(f"{len(sites)} stations x {len(years)} years, model {args.model}, lead {LEAD}")
    session = requests.Session()
    frames = []
    for site in sites:
        for y in years:
            s = max(args.start, f"{y}-01-01"); e = min(args.end, f"{y}-12-31")
            df = fetch(session, site, s, e, args.model)
            df.insert(0, "site_id", site["site_id"])
            frames.append(df)
            print(f"  {site['site_name']:<20} {y}: {df['wind_speed_10m'].notna().mean():.0%} of hours present")
            time.sleep(1.0)
    out = pd.concat(frames, ignore_index=True)
    out["model"] = args.model
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    cover = out.groupby(out["time"].dt.year)[VARIABLES].apply(lambda g: g.notna().mean()).round(3)
    print("\nShare of hours present, by year:\n" + cover.to_string())
    if out["wind_speed_10m"].notna().mean() < 0.5:
        sys.exit("Less than half of the hours have data; check the model and lead time.")
    print(f"\nWritten {OUT_PATH} ({len(out):,} rows)")


if __name__ == "__main__":
    main()