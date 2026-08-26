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
    """
    Main function that:
    1. Retrieves available monitoring sites.
    2. Checks PM2.5 data availability for selected sites.
    3. Determines the earliest available year.
    4. Calculates the length of available history.
    5. Creates a Markdown report summarising the results.
    """
    Path("docs").mkdir(exist_ok=True)

     # Retrieve all monitoring sites from the NSW Air Quality API
    sites = get_sites()
      # Display the total number of monitoring sites available
    print(f"Network: {len(sites)} sites\n")

    # If no sites have been specified for testing,
    # display all available sites so suitable sites can be selected.
    if not PROBE_SITES:
         # Loop through every monitoring site
        for s in sites:
            print(f"  {str(s.get('Site_Id')):>6}  {s.get('SiteName')}  ({s.get('Region')})")
            # Provide instructions for selecting sites
        print("\nPick 3-4 sites, add IDs to PROBE_SITES, re-run.")
        return
    
    # Create an empty list to store data-availability results
    rows = []
     # Analyse each selected monitoring site
    for sid in PROBE_SITES:
        # Find the site name that corresponds to the selected site ID.
        # If the site ID cannot be found, use the ID itself as the name.
        name = next((s.get("SiteName") for s in sites if s.get("Site_Id") == sid), str(sid))
         # Determine the earliest year containing PM2.5 data
        yr = earliest_year(sid, PARAMETER)
         # Calculate the number of years of available history
        # from the earliest year through the latest year.
        span = (LATEST_YEAR - yr) if yr else 0
         # Store the site results for later reporting
        rows.append((sid, name, yr, span))

        # Display the results for the current site
        print(f"{name:<30} {PARAMETER} from {yr or 'no data'}  ({span} yrs)")

 # Find the shortest available data history among the selected sites.
 # Sites without usable data are excluded from this calculation.
    shortest = min((r[3] for r in rows if r[2]), default=0)

     # Create a Markdown file containing the data-availability results
    with open("docs/data_availability.md", "w") as f:
        f.write(f"# Data availability - {PARAMETER}\n\n")

         # Record the date on which the API was checked
        f.write(f"Checked {date.today().isoformat()} against the NSW Air Quality API.\n\n")

         # Report the total number of monitoring sites
        f.write(f"Network size: {len(sites)} sites.\n\n")

         # Create a Markdown table containing the results
        f.write("| Site ID | Station | First year | Years of history |\n|---|---|---|---|\n")

        # Add one row to the table for each selected monitoring site
        for sid, name, yr, span in rows:
            f.write(f"| {sid} | {name} | {yr or 'none'} | {span} |\n")

        # Report the shortest usable history across the selected sites
        f.write(f"\n**Shortest usable history: {shortest} years.**\n\n")
        f.write("Note: full data validation is complete to 30 June 2022; later "
                "records have passed automated validation only.\n")
# Confirm that the Markdown report has been successfully created
    print(f"\nWritten to docs/data_availability.md")
    # Tell the user the number of years that can be used in the proposal
    print(f"Insert {shortest} years into proposal Section 4.")

# Run the main function only when this Python file is executed directly.
# This prevents main() from running automatically if the file is imported
# as a module into another Python program.
if __name__ == "__main__":
    main()
