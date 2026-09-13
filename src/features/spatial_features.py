"""
Spatial features: predict at a location using only surrounding stations.

Goal: for every (timestamp, station) pair, derive a set of features that
summarise what NEARBY stations were reporting at that time — without ever
looking at the target station's own PM2.5 value. This is what lets a model
trained on these features generalise to a brand-new location that has no
historical sensor data of its own (pure spatial interpolation/prediction).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_DIR = Path("data/processed/features")          # input: per-station time series + existing (non-spatial) features
OUT_DIR = Path("data/processed/features_spatial")        # output: same data + new spatial_* columns
SITES_FILE = Path("data/raw/sites.json")                 # station metadata (lat/lon etc.)

EARTH_RADIUS_KM = 6371.0                                  # used for haversine great-circle distance


def load_coordinates():
    """
    Load station coordinates from sites.json.

    Returns a DataFrame with one row per station: site_id, latitude, longitude.
    Stations missing a latitude (bad/incomplete metadata) are silently dropped.
    """
    with open(SITES_FILE) as f:
        sites = json.load(f)
    return pd.DataFrame([{
        "site_id": s["Site_Id"],
        "latitude": s.get("Latitude"),
        "longitude": s.get("Longitude"),
    } for s in sites if s.get("Latitude") is not None])


def haversine_matrix(coords):
    """
    Pairwise great-circle distances in km between all stations.

    Vectorised over all station pairs at once using broadcasting:
    lat[:, None] is a column vector, lat[None, :] is a row vector, so
    subtracting them gives an (S, S) matrix of all pairwise differences
    in a single operation (no Python-level double loop).

    Returns a DataFrame indexed and columned by site_id, so you can look
    up distance(i, j) as dist.loc[i, j].
    """
    lat = np.radians(coords["latitude"].values)
    lon = np.radians(coords["longitude"].values)

    dlat = lat[:, None] - lat[None, :]                    # (S, S) pairwise lat differences
    dlon = lon[:, None] - lon[None, :]                    # (S, S) pairwise lon differences

    # Standard haversine formula
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2)
    # clip to [0, 1] before sqrt/arcsin to guard against tiny floating-point
    # overshoot (e.g. 1.0000000002) which would otherwise throw a domain error
    d = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    return pd.DataFrame(d, index=coords["site_id"].values,
                        columns=coords["site_id"].values)


def bearing_matrix(coords):
    """
    Initial bearing in degrees (0-360, compass convention) from station i to
    station j. bear.loc[i, j] = the compass direction you'd travel from i to
    reach j. Used later to determine which stations sit "upwind" of a target.
    """
    lat = np.radians(coords["latitude"].values)
    lon = np.radians(coords["longitude"].values)

    dlon = lon[None, :] - lon[:, None]
    y = np.sin(dlon) * np.cos(lat)[None, :]
    x = (np.cos(lat)[:, None] * np.sin(lat)[None, :]
         - np.sin(lat)[:, None] * np.cos(lat)[None, :] * np.cos(dlon))
    # atan2 gives -180..180; shift to the usual 0..360 compass bearing range
    brng = (np.degrees(np.arctan2(y, x)) + 360) % 360

    return pd.DataFrame(brng, index=coords["site_id"].values,
                        columns=coords["site_id"].values)


def build_spatial(df, dist, bear, k, idw_power):
    """
    For each timestamp, compute neighbour-derived features per station.

    Parameters
    ----------
    df : long-format DataFrame with columns [timestamp, site_id, PM2.5, WDR, ...]
    dist, bear : (S, S) distance / bearing DataFrames from the functions above
    k : how many nearest neighbours to keep as individual features
    idw_power : exponent in the inverse-distance-weighting (higher = weight
                drops off faster with distance, i.e. more "local")
    """
    sites = sorted(df["site_id"].unique())
    # Reorder the distance/bearing matrices to match the site order we'll use
    # everywhere below, so matrix column j always corresponds to sites[j].
    dist = dist.loc[sites, sites]
    bear = bear.loc[sites, sites]

    # Reshape from long format (one row per station-timestamp) to wide format:
    # rows = timestamps, columns = stations. This lets us do fast matrix math
    # instead of grouping/looping over timestamps.
    pm = df.pivot_table(index="timestamp", columns="site_id",
                        values="PM2.5", aggfunc="first").reindex(columns=sites)
    wdr = df.pivot_table(index="timestamp", columns="site_id",
                         values="WDR", aggfunc="first").reindex(columns=sites)

    times = pm.index
    V = pm.values                                    # (T, S) PM2.5 readings, NaN where missing
    D = dist.values                                  # (S, S) distances in km
    B = bear.values                                  # (S, S) bearings in degrees
    S = len(sites)

    # --- guard against self-reference ---------------------------------
    # Set each station's distance to itself to NaN so it can never be
    # picked as its own "nearest neighbour" or contribute to its own
    # IDW/aggregate features. This is the key anti-leakage step.
    self_mask = np.eye(S, dtype=bool)
    D_safe = np.where(self_mask, np.nan, D)

    # Inverse-distance weights: closer stations => larger weight.
    # 1/0^power would be inf, but since self-distance is NaN, that produces
    # NaN here, which we immediately zero out below (self gets zero weight).
    with np.errstate(divide="ignore"):
        W = 1.0 / np.power(D_safe, idw_power)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0)        # (S, S) weight matrix, self-weight = 0

    valid = ~np.isnan(V)                             # (T, S) boolean mask of observed readings
    V0 = np.nan_to_num(V, nan=0.0)                   # missing readings treated as 0 for the matmul,
                                                      # but excluded properly via `valid` in the denominator

    # --- IDW mean of neighbours ----------------------------------------
    # For station j: idw[t, j] = sum_i( W[j,i] * V[t,i] ) / sum_i( W[j,i] ) over
    # stations i that had a valid reading at time t. Doing this as a single
    # matrix multiply (V0 @ W.T) computes it for every station at every
    # timestamp in one shot.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        num = V0 @ W.T                               # (T, S) weighted sum of neighbour values
        den = valid.astype(float) @ W.T              # (T, S) sum of weights of neighbours that had data
    idw = np.where(den > 0, num / np.where(den == 0, 1, den), np.nan)
    # (the np.where(den==0,1,den) just avoids a 0/0 warning; result is
    #  overridden to NaN by the outer np.where wherever den was actually 0)

    # --- network-wide aggregates, self excluded -------------------------
    tot = np.nansum(V, axis=1, keepdims=True)         # (T, 1) sum over ALL stations, ignoring NaN
    cnt = valid.sum(axis=1, keepdims=True)            # (T, 1) count of stations with data
    others_sum = tot - np.nan_to_num(V)               # subtract station j's own value -> sum of "others"
    others_cnt = cnt - valid.astype(int)              # subtract 1 if station j itself had data
    net_mean = np.where(others_cnt > 0,
                        others_sum / np.where(others_cnt == 0, 1, others_cnt),
                        np.nan)

    # net_max / net_std need "all other stations" per station j, which isn't
    # a simple sum, so these are computed with an explicit per-station loop
    # (S iterations, not T*S — still cheap).
    net_max = np.empty_like(V)
    net_std = np.empty_like(V)
    for j in range(S):
        others = np.delete(V, j, axis=1)              # drop column j -> everyone except station j
        net_max[:, j] = np.nanmax(others, axis=1)
        net_std[:, j] = np.nanstd(others, axis=1)

    # --- k nearest neighbours --------------------------------------------
    # For each station, sort all other stations by distance (NaN/self pushed
    # to the end via np.inf) and take the first k indices.
    order = np.argsort(np.where(np.isnan(D_safe), np.inf, D_safe), axis=1)
    knn_idx = order[:, :k]                           # (S, k) column indices of the k nearest stations

    # Gather the actual values/distances for those neighbours.
    # knn_vals[t, j, n] = PM2.5 at time t for the n-th nearest neighbour of station j
    knn_vals = np.stack([V[:, knn_idx[j]] for j in range(S)], axis=1)   # (T, S, k)
    knn_dist = np.stack([D[j, knn_idx[j]] for j in range(S)], axis=0)   # (S, k) - static, doesn't vary by time

    # --- upwind-weighted mean --------------------------------------------
    # Idea: pollution tends to travel downwind, so a neighbour is more
    # informative if it sits upwind of the target station. For station j,
    # compare each neighbour i's bearing (B[j,i]) to the wind direction at j
    # (WDR is "direction wind is coming FROM"). If they roughly line up,
    # cos(angle) is close to 1 (strongly upwind); if the neighbour is
    # downwind or perpendicular, cos(angle) <= 0 and gets weight 0.
    WD = wdr.values                                  # (T, S) wind direction per station per time
    upwind = np.full_like(V, np.nan, dtype=float)
    for j in range(S):
        ang = np.radians(B[j, :][None, :] - WD[:, [j]])   # (T, S) angle between neighbour bearing and wind-from direction
        w = np.cos(ang)                                    # +1 = directly upwind, -1 = directly downwind
        w = np.where(w > 0, w, 0.0)                         # zero out downwind/perpendicular neighbours
        w[:, j] = 0.0                                       # station never weights itself
        wv = np.nan_to_num(V, nan=0.0) * w
        wsum = (valid * w).sum(axis=1)
        upwind[:, j] = np.where(wsum > 0, wv.sum(axis=1) / np.where(wsum == 0, 1, wsum), np.nan)

    # --- assemble long-format output --------------------------------------
    frames = []
    for j, sid in enumerate(sites):
        f = pd.DataFrame({"timestamp": times, "site_id": sid})
        f["spatial_idw"] = idw[:, j]
        f["spatial_net_mean"] = net_mean[:, j]
        f["spatial_net_max"] = net_max[:, j]
        f["spatial_net_std"] = net_std[:, j]
        f["spatial_upwind"] = upwind[:, j]
        for n in range(k):
            f[f"spatial_knn{n+1}_value"] = knn_vals[:, j, n]   # n-th nearest neighbour's reading (time-varying)
            f[f"spatial_knn{n+1}_dist"] = knn_dist[j, n]       # n-th nearest neighbour's distance (constant per station)
        f["spatial_nearest_dist"] = knn_dist[j, 0]
        frames.append(f)

    return pd.concat(frames, ignore_index=True)


def check_no_self_reference(merged, dist):
    """
    Sanity check: the IDW estimate should be correlated with, but clearly
    NOT identical to, the station's own PM2.5. A correlation above 0.98
    would suggest the target station leaked into its own feature (e.g. a
    masking bug), so this flags it rather than trusting the pipeline blindly.
    """
    sub = merged.dropna(subset=["spatial_idw", "PM2.5"])
    c = sub["spatial_idw"].corr(sub["PM2.5"])
    print(f"\nSelf-reference check")
    print(f"  corr(spatial_idw, own PM2.5) = {c:.3f}")
    if c > 0.98:
        print("  WARNING: spatial feature may include the target station")
    else:
        print("  OK — spatial estimate is independent of the target sensor")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=3,
                    help="Number of nearest neighbours (default 3)")
    ap.add_argument("--idw-power", type=float, default=2.0,
                    help="Inverse distance weighting exponent (default 2)")
    args = ap.parse_args()

    print("Loading features...")
    df = pd.read_parquet(FEATURES_DIR)
    print(f"  {len(df):,} rows, {df['site_id'].nunique()} stations")

    coords = load_coordinates()
    coords = coords[coords["site_id"].isin(df["site_id"].unique())]   # keep only stations present in the feature data
    print(f"  coordinates for {len(coords)} stations")

    dist = haversine_matrix(coords)
    bear = bearing_matrix(coords)

    # Quick diagnostic: how far apart are stations, roughly?
    # This helps sanity-check whether k and idw_power are sensible for the
    # network's geographic density (e.g. a huge min/max spread might mean
    # some stations are effectively isolated).
    d = dist.values[~np.eye(len(coords), dtype=bool)]
    print(f"\nStation separation (km)")
    print(f"  min {d.min():.1f}   median {np.median(d):.1f}   max {d.max():.1f}")

    print(f"\nBuilding spatial features (k={args.k}, IDW power={args.idw_power})...")
    spatial = build_spatial(df, dist, bear, args.k, args.idw_power)
    print(f"  {len(spatial):,} station-hours")

    # Merge the new spatial_* columns back onto the original feature table,
    # plus attach the station coordinates themselves (handy for later
    # modelling/plotting).
    merged = df.merge(spatial, on=["timestamp", "site_id"], how="left")
    merged = merged.merge(coords, on="site_id", how="left")

    check_no_self_reference(merged, dist)

    # Report non-null coverage for a sample of the new columns, so you can
    # spot at a glance if something (e.g. wind data) is mostly missing.
    spatial_cols = [c for c in merged.columns if c.startswith("spatial_")]
    print(f"\nSpatial feature coverage")
    for c in spatial_cols[:6]:
        print(f"  {c:<26} {merged[c].notna().mean():.1%} non-null")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged["year"] = merged["timestamp"].dt.year
    merged.to_parquet(OUT_DIR, partition_cols=["year"], index=False)   # partitioned by year for efficient downstream reads

    print(f"\nWritten to {OUT_DIR}/")
    print(f"  rows             : {len(merged):,}")
    print(f"  spatial features : {len(spatial_cols)}")
    print(f"  total features   : {len(merged.columns)}")


if __name__ == "__main__":
    main()