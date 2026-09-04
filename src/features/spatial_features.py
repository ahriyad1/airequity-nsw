"""
Spatial features: predict at a location using only surrounding stations.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_DIR = Path("data/processed/features")
OUT_DIR = Path("data/processed/features_spatial")
SITES_FILE = Path("data/raw/sites.json")

EARTH_RADIUS_KM = 6371.0


def load_coordinates():
    with open(SITES_FILE) as f:
        sites = json.load(f)
    return pd.DataFrame([{
        "site_id": s["Site_Id"],
        "latitude": s.get("Latitude"),
        "longitude": s.get("Longitude"),
    } for s in sites if s.get("Latitude") is not None])


def haversine_matrix(coords):
    """Pairwise great-circle distances in km between all stations."""
    lat = np.radians(coords["latitude"].values)
    lon = np.radians(coords["longitude"].values)

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2)
    d = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    return pd.DataFrame(d, index=coords["site_id"].values,
                        columns=coords["site_id"].values)


def bearing_matrix(coords):
    """
    Initial bearing in degrees from station i to station j.
    """
    lat = np.radians(coords["latitude"].values)
    lon = np.radians(coords["longitude"].values)

    dlon = lon[None, :] - lon[:, None]
    y = np.sin(dlon) * np.cos(lat)[None, :]
    x = (np.cos(lat)[:, None] * np.sin(lat)[None, :]
         - np.sin(lat)[:, None] * np.cos(lat)[None, :] * np.cos(dlon))
    brng = (np.degrees(np.arctan2(y, x)) + 360) % 360

    return pd.DataFrame(brng, index=coords["site_id"].values,
                        columns=coords["site_id"].values)


def build_spatial(df, dist, bear, k, idw_power):
    """
    For each timestamp, compute neighbour-derived features per station.
    """
    sites = sorted(df["site_id"].unique())
    dist = dist.loc[sites, sites]
    bear = bear.loc[sites, sites]

    # Wide matrix: rows = timestamps, columns = stations, values = PM2.5
    pm = df.pivot_table(index="timestamp", columns="site_id",
                        values="PM2.5", aggfunc="first").reindex(columns=sites)
    wdr = df.pivot_table(index="timestamp", columns="site_id",
                         values="WDR", aggfunc="first").reindex(columns=sites)

    times = pm.index
    V = pm.values                                    # (T, S)
    D = dist.values                                  # (S, S)
    B = bear.values                                  # (S, S)
    S = len(sites)

    # Mask self-comparisons permanently
    self_mask = np.eye(S, dtype=bool)
    D_safe = np.where(self_mask, np.nan, D)

    # IDW weights, self excluded
    with np.errstate(divide="ignore"):
        W = 1.0 / np.power(D_safe, idw_power)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0)

    valid = ~np.isnan(V)                             # (T, S)
    V0 = np.nan_to_num(V, nan=0.0)

    # --- IDW mean of neighbours -------------
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        num = V0 @ W.T                               # (T, S)
        den = valid.astype(float) @ W.T
    idw = np.where(den > 0, num / np.where(den == 0, 1, den), np.nan)

    # --- network aggregates, self excluded --------------
    tot = np.nansum(V, axis=1, keepdims=True)
    cnt = valid.sum(axis=1, keepdims=True)
    others_sum = tot - np.nan_to_num(V)
    others_cnt = cnt - valid.astype(int)
    net_mean = np.where(others_cnt > 0,
                        others_sum / np.where(others_cnt == 0, 1, others_cnt),
                        np.nan)

    net_max = np.empty_like(V)
    net_std = np.empty_like(V)
    for j in range(S):
        others = np.delete(V, j, axis=1)
        net_max[:, j] = np.nanmax(others, axis=1)
        net_std[:, j] = np.nanstd(others, axis=1)

    # --- k nearest neighbours ---------------------------------------
    order = np.argsort(np.where(np.isnan(D_safe), np.inf, D_safe), axis=1)
    knn_idx = order[:, :k]                           # (S, k)

    knn_vals = np.stack([V[:, knn_idx[j]] for j in range(S)], axis=1)
    knn_dist = np.stack([D[j, knn_idx[j]] for j in range(S)], axis=0)

    # --- upwind weighting -------------------------------------------
    # cos of the angle between wind origin and neighbour bearing; only
    # neighbours in the upwind half-plane contribute.
    WD = wdr.values                                  # (T, S)
    upwind = np.full_like(V, np.nan, dtype=float)
    for j in range(S):
        ang = np.radians(B[j, :][None, :] - WD[:, [j]])
        w = np.cos(ang)
        w = np.where(w > 0, w, 0.0)
        w[:, j] = 0.0
        wv = np.nan_to_num(V, nan=0.0) * w
        wsum = (valid * w).sum(axis=1)
        upwind[:, j] = np.where(wsum > 0, wv.sum(axis=1) / np.where(wsum == 0, 1, wsum), np.nan)

    # --- assemble ----------------------------------------------------
    frames = []
    for j, sid in enumerate(sites):
        f = pd.DataFrame({"timestamp": times, "site_id": sid})
        f["spatial_idw"] = idw[:, j]
        f["spatial_net_mean"] = net_mean[:, j]
        f["spatial_net_max"] = net_max[:, j]
        f["spatial_net_std"] = net_std[:, j]
        f["spatial_upwind"] = upwind[:, j]
        for n in range(k):
            f[f"spatial_knn{n+1}_value"] = knn_vals[:, j, n]
            f[f"spatial_knn{n+1}_dist"] = knn_dist[j, n]
        f["spatial_nearest_dist"] = knn_dist[j, 0]
        frames.append(f)

    return pd.concat(frames, ignore_index=True)


def check_no_self_reference(merged, dist):
    """
    Confirm spatial features are not simply the station's own reading.
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
    coords = coords[coords["site_id"].isin(df["site_id"].unique())]
    print(f"  coordinates for {len(coords)} stations")

    dist = haversine_matrix(coords)
    bear = bearing_matrix(coords)

    d = dist.values[~np.eye(len(coords), dtype=bool)]
    print(f"\nStation separation (km)")
    print(f"  min {d.min():.1f}   median {np.median(d):.1f}   max {d.max():.1f}")

    print(f"\nBuilding spatial features (k={args.k}, IDW power={args.idw_power})...")
    spatial = build_spatial(df, dist, bear, args.k, args.idw_power)
    print(f"  {len(spatial):,} station-hours")

    merged = df.merge(spatial, on=["timestamp", "site_id"], how="left")
    merged = merged.merge(coords, on="site_id", how="left")

    check_no_self_reference(merged, dist)

    spatial_cols = [c for c in merged.columns if c.startswith("spatial_")]
    print(f"\nSpatial feature coverage")
    for c in spatial_cols[:6]:
        print(f"  {c:<26} {merged[c].notna().mean():.1%} non-null")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged["year"] = merged["timestamp"].dt.year
    merged.to_parquet(OUT_DIR, partition_cols=["year"], index=False)

    print(f"\nWritten to {OUT_DIR}/")
    print(f"  rows             : {len(merged):,}")
    print(f"  spatial features : {len(spatial_cols)}")
    print(f"  total features   : {len(merged.columns)}")


if __name__ == "__main__":
    main()