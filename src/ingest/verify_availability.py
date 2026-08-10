import json
from datetime import date
from pathlib import Path
import requests

BASE = "https://data.airquality.nsw.gov.au"
SITES_URL = f"{BASE}/api/Data/get_SiteDetails"
PARAMS_URL = f"{BASE}/api/Data/get_ParameterDetails"
OBS_URL = f"{BASE}/api/Data/get_Observations"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
TIMEOUT = 90

PROBE_SITES = [39, 107, 171, 650]
PARAMETER = "PM2.5"
EARLIEST_YEAR = 1994
LATEST_YEAR = date.today().year


def get_sites():
    r = requests.get(SITES_URL, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def has_data(site_id, parameter, year):
    payload = {
        "Parameters": [parameter],
        "Sites": [site_id],
        "StartDate": f"{year}-01-01",
        "EndDate": f"{year}-12-31",
        "Categories": ["Averages"],
        "SubCategories": ["Hourly"],
    }
    r = requests.post(OBS_URL, headers=HEADERS, data=json.dumps(payload), timeout=TIMEOUT)
    r.raise_for_status()
    body = r.json()
    records = body.get("Values", []) if isinstance(body, dict) else body
    return any(rec.get("Value") is not None for rec in records)


def earliest_year(site_id, parameter):
    lo, hi = EARLIEST_YEAR, LATEST_YEAR
    if not has_data(site_id, parameter, hi):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if has_data(site_id, parameter, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def main():
    Path("docs").mkdir(exist_ok=True)
    sites = get_sites()
    print(f"Network: {len(sites)} sites\n")

    if not PROBE_SITES:
        for s in sites:
            print(f"  {str(s.get('Site_Id')):>6}  {s.get('SiteName')}  ({s.get('Region')})")
        print("\nPick 3-4 sites, add IDs to PROBE_SITES, re-run.")
        return

    rows = []
    for sid in PROBE_SITES:
        name = next((s.get("SiteName") for s in sites if s.get("Site_Id") == sid), str(sid))
        yr = earliest_year(sid, PARAMETER)
        span = (LATEST_YEAR - yr) if yr else 0
        rows.append((sid, name, yr, span))
        print(f"{name:<30} {PARAMETER} from {yr or 'no data'}  ({span} yrs)")

    shortest = min((r[3] for r in rows if r[2]), default=0)
    with open("docs/data_availability.md", "w") as f:
        f.write(f"# Data availability - {PARAMETER}\n\n")
        f.write(f"Checked {date.today().isoformat()} against the NSW Air Quality API.\n\n")
        f.write(f"Network size: {len(sites)} sites.\n\n")
        f.write("| Site ID | Station | First year | Years of history |\n|---|---|---|---|\n")
        for sid, name, yr, span in rows:
            f.write(f"| {sid} | {name} | {yr or 'none'} | {span} |\n")
        f.write(f"\n**Shortest usable history: {shortest} years.**\n\n")
        f.write("Note: full data validation is complete to 30 June 2022; later "
                "records have passed automated validation only.\n")

    print(f"\nWritten to docs/data_availability.md")
    print(f"Insert {shortest} years into proposal Section 4.")


if __name__ == "__main__":
    main()
