"""
Fetch air quality observations from the NSW Air Quality API

Card: AIR-9 — Build NSW Air Quality API ingestion script
"""

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

BASE = "https://data.airquality.nsw.gov.au"
SITES_URL = f"{BASE}/api/Data/get_SiteDetails"
PARAMS_URL = f"{BASE}/api/Data/get_ParameterDetails"
OBS_URL = f"{BASE}/api/Data/get_Observations"

HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
TIMEOUT = 120
MAX_RETRIES = 4
# the API times out on long ranges
CHUNK_DAYS = 30   
# the API times out on long ranges       
PAUSE_SECONDS = 1.0     

RAW_DIR = Path("data/raw")

# Sydney monitoring regions. Region-level aggregates and test sites are
# excluded by filter_real_stations() below.
SYDNEY_REGIONS = {"Sydney East", "Sydney South-west", "Sydney North-west"}

DEFAULT_PARAMETERS = ["PM2.5", "PM10", "OZONE", "NO2",
                      "TEMP", "HUMID", "WSP", "WDR", "SOLAR", "RAIN"]


# API helpers

def _request(method, url, payload=None):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if method == "GET":
                r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            else:
                r = requests.post(url, headers=HEADERS,
                                  data=json.dumps(payload), timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            print(f"    attempt {attempt} failed ({type(e).__name__}), "
                  f"retrying in {wait}s", flush=True)
            time.sleep(wait)


def get_sites():
    return _request("GET", SITES_URL)


def get_parameters():
    return _request("GET", PARAMS_URL)


def filter_real_stations(sites, regions=None):
    out = []
    for s in sites:
        sid = s.get("Site_Id")
        name = (s.get("SiteName") or "").strip()
        region = (s.get("Region") or "").strip()

        if sid is None or sid > 1_000_000:
            continue
        if "test" in name.lower() or "test" in region.lower():
            continue
        if regions and region not in regions:
            continue
        out.append(s)
    return out


def validate_records(records):
    if not isinstance(records, list):
        raise ValueError(f"expected a list, got {type(records).__name__}")
    if not records:
        return 0
    required = {"Site_Id", "Parameter", "Date", "Hour", "Value"}
    missing = required - set(records[0].keys())
    if missing:
        raise ValueError(f"response missing expected fields: {missing}")
    return len(records)


# Fetching

def date_chunks(start, end, days=CHUNK_DAYS):
    #Split a date range into chunks
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=days - 1), end)
        yield cur, stop
        cur = stop + timedelta(days=1)


def fetch_chunk(site_ids, parameters, start, end):
    payload = {
        "Parameters": parameters,
        "Sites": site_ids,
        "StartDate": start.isoformat(),
        "EndDate": end.isoformat(),
        "Categories": ["Averages"],
        "SubCategories": ["Hourly"],
    }
    body = _request("POST", OBS_URL, payload)
    records = body.get("Values", []) if isinstance(body, dict) else body
    return records


def save_raw(records, start, end, tag):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fname = RAW_DIR / f"obs_{tag}_{start.isoformat()}_{end.isoformat()}.json"
    with open(fname, "w") as f:
        json.dump(records, f)
    return fname


# Main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--sites", nargs="*", type=int,
                    help="Site IDs. Omit for all Sydney stations.")
    ap.add_argument("--parameters", nargs="*", default=DEFAULT_PARAMETERS)
    ap.add_argument("--batch", type=int, default=10,
                    help="Sites per request (default 10)")
    ap.add_argument("--list-sites", action="store_true",
                    help="Print Sydney stations and exit")
    ap.add_argument("--list-parameters", action="store_true",
                    help="Print available parameters and exit")
    args = ap.parse_args()

    if args.list_parameters:
        for p in get_parameters():
            print(f"  {p.get('ParameterCode'):<10} {p.get('ParameterDescription')}")
        return

    sites = get_sites()
    sydney = filter_real_stations(sites, SYDNEY_REGIONS)

    if args.list_sites:
        print(f"{len(sydney)} Sydney monitoring stations "
              f"(from {len(sites)} API entries)\n")
        for s in sorted(sydney, key=lambda x: x.get("Region") or ""):
            print(f"  {str(s.get('Site_Id')):>6}  {s.get('SiteName'):<24} "
                  f"{s.get('Region')}")
        return

    if not args.start or not args.end:
        ap.error("--start and --end are required (or use --list-sites)")

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if end < start:
        ap.error("--end must not be before --start")

    site_ids = args.sites or [s["Site_Id"] for s in sydney]
    print(f"Fetching {len(args.parameters)} parameters for "
          f"{len(site_ids)} sites, {start} to {end}\n")

    total = 0
    failures = []

    for batch_no, i in enumerate(range(0, len(site_ids), args.batch), start=1):
        batch = site_ids[i:i + args.batch]
        for c_start, c_end in date_chunks(start, end):
            label = f"batch {batch_no}, {c_start} to {c_end}"
            try:
                records = fetch_chunk(batch, args.parameters, c_start, c_end)
                n = validate_records(records)
                if n:
                    path = save_raw(records, c_start, c_end, f"b{batch_no}")
                    print(f"  {label}: {n:,} records -> {path.name}")
                else:
                    print(f"  {label}: no data")
                total += n
            except Exception as e:
                print(f"  {label}: FAILED — {e}")
                failures.append(label)
            time.sleep(PAUSE_SECONDS)

    print(f"\nTotal records: {total:,}")
    print(f"Saved to: {RAW_DIR}/")
    if failures:
        print(f"\n{len(failures)} chunk(s) failed:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()