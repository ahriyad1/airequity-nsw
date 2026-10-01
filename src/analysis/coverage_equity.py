"""
Coverage and equity: who lives near a PM2.5 monitor, and does that track
socio-economic disadvantage?

DATA:
    https://geo.abs.gov.au/arcgis/rest/services/Hosted/
        ABS_Socio_Economic_Indexes_for_Areas_SEIFA_by_2021_SA2/FeatureServer/0

    The raw response is archived to data/raw/ for lineage.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

SEIFA_SERVICE = ("https://geo.abs.gov.au/arcgis/rest/services/Hosted/"
                 "ABS_Socio_Economic_Indexes_for_Areas_SEIFA_by_2021_SA2/"
                 "FeatureServer/0/query")
SA2_FIELDS = ["sa2_code_2021", "sa2_name_2021", "area_albers_sqkm", "urp",
              "irsd_score", "irsd_state_decile", "irsd_aus_decile"]
SYDNEY_BASIN_SA4 = ("115000000", "129000000")    # SA4 115 to 128 inclusive

RAW_SA2 = Path("data/raw/abs_seifa_sa2_sydney.json")
SITES_FILE = Path("data/raw/sites.json")
FEATURES_DIR = Path("data/processed/features")
RESULTS_DIR = Path("results")
LOSO_PRECISION = RESULTS_DIR / "bootstrap_detail_precision.csv"
LOSO_RECALL = RESULTS_DIR / "bootstrap_detail_recall.csv"

DISTANCE_BANDS_KM = [5, 10, 20]
EARTH_RADIUS_KM = 6371.0
TIMEOUT = 90


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _query(params):
    r = requests.get(SEIFA_SERVICE, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise RuntimeError(f"ABS service error: {body['error']}")
    return body


def fetch_sa2(refresh=False):
   #Sydney basin SA2s with population, area, IRSD and centroid
    if RAW_SA2.exists() and not refresh:
        records = json.loads(RAW_SA2.read_text())
        print(f"  using cached ABS data ({len(records)} SA2s) — --refresh to re-download")
        return pd.DataFrame(records)

    where = (f"sa2_code_2021 >= '{SYDNEY_BASIN_SA4[0]}' "
             f"AND sa2_code_2021 < '{SYDNEY_BASIN_SA4[1]}'")
    base = {"where": where, "outFields": ",".join(SA2_FIELDS),
            "outSR": "4326", "f": "json", "resultRecordCount": 2000}

    features, offset = [], 0
    while True:
        body = _query({**base, "returnGeometry": "false", "returnCentroid": "true",
                       "resultOffset": offset})
        features += body.get("features", [])
        if not body.get("exceededTransferLimit"):
            break
        offset += len(body.get("features", []))

    records = []
    need_geometry = []
    for f in features:
        rec = dict(f["attributes"])
        c = f.get("centroid")
        if c and c.get("x") is not None:
            rec["longitude"], rec["latitude"] = c["x"], c["y"]
        else:
            need_geometry.append(rec["sa2_code_2021"])
        records.append(rec)

    if need_geometry:
        # Fallback: compute an approximate centroid from the outer ring
        print(f"  service returned no centroid for {len(need_geometry)} SA2s — "
              f"computing from boundaries")
        body = _query({**base, "returnGeometry": "true", "geometryPrecision": 5,
                       "outFields": "sa2_code_2021"})
        approx = {}
        for f in body.get("features", []):
            rings = (f.get("geometry") or {}).get("rings") or []
            if rings:
                ring = np.array(max(rings, key=len))
                approx[f["attributes"]["sa2_code_2021"]] = ring.mean(axis=0)
        for rec in records:
            if "latitude" not in rec and rec["sa2_code_2021"] in approx:
                rec["longitude"], rec["latitude"] = approx[rec["sa2_code_2021"]]

    RAW_SA2.parent.mkdir(parents=True, exist_ok=True)
    RAW_SA2.write_text(json.dumps(records, indent=1))
    print(f"  downloaded {len(records)} SA2s from the ABS, archived to {RAW_SA2}")
    return pd.DataFrame(records)


def load_stations():
    #Operational stations with coordinates, region and crossing rate
    feat = pd.read_parquet(FEATURES_DIR, columns=["site_id", "site_name", "region", "label"])
    feat = feat.dropna(subset=["label"])
    rates = (feat.groupby(["site_id", "site_name", "region"], as_index=False)["label"]
                 .mean().rename(columns={"label": "crossing_rate"}))
    sites = json.loads(SITES_FILE.read_text())
    coords = pd.DataFrame([{"site_id": s["Site_Id"], "latitude": s.get("Latitude"),
                            "longitude": s.get("Longitude")} for s in sites])
    return rates.merge(coords, on="site_id", how="left").dropna(subset=["latitude"])


def haversine_matrix(lat1, lon1, lat2, lon2):
    """Distances in km between every point in set 1 and every point in set 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat1[:, None] - lat2[None, :]
    dlon = lon1[:, None] - lon2[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1)[:, None] * np.cos(lat2)[None, :] * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def weighted_mean(values, weights):
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    ok = ~np.isnan(v) & ~np.isnan(w) & (w > 0)
    return float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan


def weighted_median(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = ~np.isnan(v) & (w > 0)
    v, w = v[ok], w[ok]
    order = np.argsort(v)
    cum = np.cumsum(w[order])
    return float(v[order][np.searchsorted(cum, cum[-1] / 2)])


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def assign_nearest(sa2, stations):
    d = haversine_matrix(sa2["latitude"].values, sa2["longitude"].values,
                         stations["latitude"].values, stations["longitude"].values)
    idx = d.argmin(axis=1)
    out = sa2.copy()
    out["nearest_km"] = d[np.arange(len(sa2)), idx]
    out["nearest_site_id"] = stations["site_id"].values[idx]
    out["nearest_station"] = stations["site_name"].values[idx]
    out["region"] = stations["region"].values[idx]
    return out


def coverage_by_region(sa2, stations):
    rows = []
    for region, g in sa2.groupby("region"):
        pop = g["urp"].sum()
        area = g["area_albers_sqkm"].sum()
        n = int((stations["region"] == region).sum())
        row = {
            "region": region, "stations": n,
            "population": int(pop), "area_km2": round(area, 1),
            "stations_per_100k": round(n / pop * 1e5, 3) if pop else np.nan,
            "stations_per_1000km2": round(n / area * 1e3, 3) if area else np.nan,
            "residents_per_station": int(pop / n) if n else np.nan,
            "pop_weighted_mean_km": round(weighted_mean(g["nearest_km"], g["urp"]), 2),
        }
        for b in DISTANCE_BANDS_KM:
            row[f"pct_pop_beyond_{b}km"] = round(
                100 * g.loc[g["nearest_km"] > b, "urp"].sum() / pop, 1) if pop else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("stations_per_100k")


def coverage_by_decile(sa2):
    rows = []
    s = sa2.dropna(subset=["irsd_state_decile"])
    for dec, g in s.groupby("irsd_state_decile"):
        pop = g["urp"].sum()
        row = {
            "irsd_nsw_decile": int(dec), "sa2_count": len(g), "population": int(pop),
            "pop_weighted_mean_km": round(weighted_mean(g["nearest_km"], g["urp"]), 2),
        }
        for b in DISTANCE_BANDS_KM:
            row[f"pct_pop_beyond_{b}km"] = round(
                100 * g.loc[g["nearest_km"] > b, "urp"].sum() / pop, 1) if pop else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("irsd_nsw_decile")


def station_catchments(sa2, stations):
    rows = []
    for site_id, g in sa2.groupby("nearest_site_id"):
        rows.append({
            "site_id": site_id,
            "catchment_population": int(g["urp"].sum()),
            "catchment_area_km2": round(float(g["area_albers_sqkm"].sum()), 1),
            "catchment_sa2s": len(g),
            "catchment_irsd": round(weighted_mean(g["irsd_score"], g["urp"]), 1),
            "catchment_mean_km": round(weighted_mean(g["nearest_km"], g["urp"]), 2),
        })
    agg = pd.DataFrame(rows)
    out = stations.merge(agg, on="site_id", how="left")
    out["catchment_population"] = out["catchment_population"].fillna(0).astype(int)

    for path, metric in [(LOSO_PRECISION, "precision"), (LOSO_RECALL, "recall")]:
        if path.exists():
            loso = pd.read_csv(path)
            keep = [c for c in ["site_name", "monitored", "unmonitored", "sensor_gap"]
                    if c in loso.columns]
            loso = loso[keep].rename(columns={
                "monitored": f"{metric}_with_sensor",
                "unmonitored": f"{metric}_without_sensor",
                "sensor_gap": f"{metric}_sensor_gap"})
            out = out.merge(loso, on="site_name", how="left")
    return out.sort_values("catchment_irsd")


def spearman(x, y):
    ok = ~(pd.isna(x) | pd.isna(y))
    if ok.sum() < 5:
        return np.nan, np.nan, int(ok.sum())
    rho, p = spearmanr(x[ok], y[ok])
    return float(rho), float(p), int(ok.sum())


def describe(rho, p):
    if np.isnan(rho):
        return "insufficient data"
    strength = ("negligible" if abs(rho) < .1 else "weak" if abs(rho) < .3
                else "moderate" if abs(rho) < .5 else "strong")
    sig = "significant" if p < .05 else "not significant"
    return f"{strength}, {sig}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help="Re-download SA2 data from the ABS instead of using the cache")
    args = ap.parse_args()

    print("Loading ABS SEIFA 2021 by SA2 (Sydney basin)...")
    sa2 = fetch_sa2(args.refresh)
    for c in ["area_albers_sqkm", "urp", "irsd_score", "irsd_state_decile",
              "irsd_aus_decile", "latitude", "longitude"]:
        sa2[c] = pd.to_numeric(sa2[c], errors="coerce")
    sa2 = sa2.dropna(subset=["latitude", "longitude"])
    sa2["urp"] = sa2["urp"].fillna(0)
    print(f"  {len(sa2)} SA2s, {int(sa2['urp'].sum()):,} residents, "
          f"{sa2['area_albers_sqkm'].sum():,.0f} km²")

    print("Loading operational stations...")
    stations = load_stations()
    print(f"  {len(stations)} stations")

    sa2 = assign_nearest(sa2, stations)
    populated = sa2[sa2["urp"] > 0]
    total_pop = populated["urp"].sum()

    # ---- coverage -----
    print("\n" + "=" * 78)
    print("COVERAGE")
    print("=" * 78)
    pw_mean = weighted_mean(populated["nearest_km"], populated["urp"])
    pw_median = weighted_median(populated["nearest_km"], populated["urp"])
    print(f"Distance from home to nearest PM2.5 monitor (population-weighted)")
    print(f"  mean {pw_mean:.1f} km, median {pw_median:.1f} km")
    beyond = {}
    for b in DISTANCE_BANDS_KM:
        share = populated.loc[populated["nearest_km"] > b, "urp"].sum() / total_pop
        beyond[b] = share
        print(f"  {share:6.1%} of residents live more than {b} km from a monitor")

    by_region = coverage_by_region(populated, stations)
    print("\nBy region")
    print(by_region[["region", "stations", "population", "residents_per_station",
                     "stations_per_100k", "stations_per_1000km2",
                     "pop_weighted_mean_km", "pct_pop_beyond_10km"]]
          .to_string(index=False))

    # ---- equity: area level ----
    print("\n" + "=" * 78)
    print("EQUITY — AREA LEVEL")
    print("=" * 78)
    by_decile = coverage_by_decile(populated)
    print("Distance to nearest monitor by IRSD decile (1 = most disadvantaged in NSW)")
    print(by_decile[["irsd_nsw_decile", "sa2_count", "population",
                     "pop_weighted_mean_km", "pct_pop_beyond_10km"]].to_string(index=False))

    rho_a, p_a, n_a = spearman(populated["irsd_score"], populated["nearest_km"])
    print(f"\nSpearman, IRSD score vs distance (n = {n_a} SA2s): "
          f"rho = {rho_a:+.3f}, p = {p_a:.4f} — {describe(rho_a, p_a)}")
    print("  Negative rho would mean more disadvantaged areas are further from monitors.")

    # ---- equity: station level ------
    print("\n" + "=" * 78)
    print(f"EQUITY — STATION LEVEL (n = {len(stations)}, indicative only)")
    print("=" * 78)
    catch = station_catchments(populated, stations)
    show = [c for c in ["site_name", "region", "catchment_population", "catchment_irsd",
                        "crossing_rate", "precision_sensor_gap", "recall_sensor_gap"]
            if c in catch.columns]
    view = catch[show].copy()
    view["crossing_rate"] = (view["crossing_rate"] * 100).round(2)
    print(view.rename(columns={"crossing_rate": "crossing_%"})
              .round(3).to_string(index=False))

    tests = {"crossing_rate": "Crossing rate"}
    if "precision_sensor_gap" in catch.columns:
        tests["precision_sensor_gap"] = "Precision penalty without sensor"
    if "recall_sensor_gap" in catch.columns:
        tests["recall_sensor_gap"] = "Recall penalty without sensor"
    station_tests = {}
    print()
    for col, label in tests.items():
        rho, p, n = spearman(catch["catchment_irsd"], catch[col])
        station_tests[col] = {"rho": rho, "p": p, "n": n}
        print(f"  catchment IRSD vs {label:<34} rho = {rho:+.3f}, p = {p:.3f} — "
              f"{describe(rho, p)}")
    print("  Negative rho would mean more disadvantaged catchments fare worse.")

    # ---- save -------
    RESULTS_DIR.mkdir(exist_ok=True)
    by_region.to_csv(RESULTS_DIR / "coverage_by_region.csv", index=False)
    by_decile.to_csv(RESULTS_DIR / "coverage_by_decile.csv", index=False)
    catch.to_csv(RESULTS_DIR / "station_catchments.csv", index=False)
    sa2[["sa2_code_2021", "sa2_name_2021", "urp", "area_albers_sqkm", "irsd_score",
         "irsd_state_decile", "nearest_station", "region", "nearest_km"]
        ].to_csv(RESULTS_DIR / "sa2_nearest_station.csv", index=False)

    summary = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "ABS SEIFA 2021 by SA2 (Digital Atlas of Australia feature service)",
        "study_area": "Sydney basin, SA4 115-128 (excludes Central Coast)",
        "population_basis": "2021 Census usual resident population",
        "n_sa2": int(len(populated)),
        "population": int(total_pop),
        "n_stations": int(len(stations)),
        "pop_weighted_mean_km": round(pw_mean, 2),
        "pop_weighted_median_km": round(pw_median, 2),
        "share_beyond_km": {str(b): round(v, 4) for b, v in beyond.items()},
        "area_level_spearman": {"rho": rho_a, "p": p_a, "n": n_a},
        "station_level_spearman": station_tests,
    }
    (RESULTS_DIR / "coverage_equity_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {RESULTS_DIR}/: coverage_by_region.csv, coverage_by_decile.csv, "
          f"station_catchments.csv, sa2_nearest_station.csv, coverage_equity_summary.json")


if __name__ == "__main__":
    main()