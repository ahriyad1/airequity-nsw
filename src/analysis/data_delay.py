"""
The cost of data delay: how much accuracy is lost when station readings
arrive late?

METHOD
    Train on 2023, test on 2024, the same temporal split as the baselines.
    One model is fitted per condition on current (undelayed) features, as in
    production. At test time, the station-derived inputs are replaced by
    their values from L hours earlier:

        delayed    spatial_* features (neighbour readings) and, in the
                   monitored condition, the station's own PM2.5 history
        unchanged  weather, calendar and location, which are available at the
                   right hour in practice
        label      PM2.5 at t + 24 hours, as always

    The same test rows are used for every delay, so differences come from
    the delay alone. Uncertainty comes from a paired bootstrap over stations:
    each resample draws stations with replacement and recomputes every delay
    on the same draw.

    Thresholds are the cost-optimal values used by the live system.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except (ImportError, OSError):
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_LGBM = False

FEATURES_DIR = Path("data/processed/features_spatial")
RESULTS_DIR = Path("results")

SPATIAL_PREFIX = "spatial_"
OWN_SENSOR_PREFIXES = ("pm25_lag_", "pm25_mean_", "pm25_max_", "pm25_std_",
                       "pm25_delta_")
OWN_SENSOR_EXACT = {"PM2.5", "imputed"}
WEATHER = ["TEMP", "HUMID", "WSP", "WDR", "wind_u", "wind_v",
           "temp_lag_24h", "humid_lag_24h", "wsp_lag_24h", "wdr_lag_24h"]
CALENDAR = ["hour", "dayofweek", "month", "is_weekend",
            "hour_sin", "hour_cos", "month_sin", "month_cos"]
GEO = ["latitude", "longitude"]

THRESHOLDS = {"monitored": 0.065, "unmonitored": 0.090}
COST_RATIO = 10.0
DEFAULT_LAGS = [0, 12, 24, 36, 48, 72]
PRODUCTION_RANGE = (16, 40)


def feature_groups(df):
    spatial = [c for c in df.columns if c.startswith(SPATIAL_PREFIX)]
    own = [c for c in df.columns
           if c.startswith(OWN_SENSOR_PREFIXES) or c in OWN_SENSOR_EXACT]
    fixed = ([c for c in WEATHER if c in df.columns]
             + [c for c in CALENDAR if c in df.columns]
             + [c for c in GEO if c in df.columns])
    return {
        "monitored": {"features": spatial + fixed + own, "delayed": spatial + own},
        "unmonitored": {"features": spatial + fixed, "delayed": spatial},
    }


def make_model():
    #Same configuration as the validated model
    if HAVE_LGBM:
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=42)
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=50, random_state=42)


def delayed_values(full, test, cols, lag_h):
   #Values of `cols` at (site, t - lag) for each test row; NaN if absent
    lookup = full.set_index(["site_id", "timestamp"])[cols]
    lookup = lookup[~lookup.index.duplicated()]
    idx = pd.MultiIndex.from_arrays(
        [test["site_id"].values, test["timestamp"].values - np.timedelta64(lag_h, "h")])
    present = idx.isin(lookup.index)
    vals = lookup.reindex(idx)
    vals.index = test.index
    return vals, present


def station_counts(sites, y, pred):
    #True positives, false positives, false negatives and rows per station
    df = pd.DataFrame({"site": sites,
                       "tp": (y == 1) & (pred == 1),
                       "fp": (y == 0) & (pred == 1),
                       "fn": (y == 1) & (pred == 0)})
    df["n"] = 1
    return df.groupby("site")[["tp", "fp", "fn", "n"]].sum()


def scores(c):
    tp, fp, fn, n = c["tp"], c["fp"], c["fn"], c["n"]
    return {
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "cost": (COST_RATIO * fn + fp) / n if n else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lags", type=int, nargs="+", default=DEFAULT_LAGS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    lags = sorted(set([0] + args.lags))
    rng = np.random.default_rng(args.seed)

    print("Loading features...")
    df = pd.read_parquet(FEATURES_DIR).dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    groups = feature_groups(df)

    train = df[df["timestamp"].dt.year == 2023]
    test = df[df["timestamp"].dt.year == 2024].copy()
    print(f"  train 2023: {len(train):,} rows | test 2024: {len(test):,} rows, "
          f"{int(test['label'].sum()):,} exceedances, {test['site_id'].nunique()} stations")

    # Rows where the delayed inputs exist for every lag, so every delay is
    # scored on exactly the same rows
    all_delayed = sorted(set(groups["monitored"]["delayed"]))
    common = np.ones(len(test), dtype=bool)
    for L in lags:
        _, present = delayed_values(df, test, all_delayed, L)
        common &= present
    test = test[common].copy()
    print(f"  {len(test):,} test rows have inputs available at every delay "
          f"({int(test['label'].sum()):,} exceedances)")

    rows, boot_store = [], {}
    for cond, spec in groups.items():
        print(f"\nFitting {cond} model ({len(spec['features'])} features)...")
        model = make_model().fit(train[spec["features"]], train["label"].values)
        thr = THRESHOLDS[cond]
        y = test["label"].values

        for L in lags:
            X = test[spec["features"]].copy()
            if L > 0:
                vals, _ = delayed_values(df, test, spec["delayed"], L)
                X[spec["delayed"]] = vals[spec["delayed"]].values
            prob = model.predict_proba(X)[:, 1]
            pred = (prob >= thr).astype(int)
            c = station_counts(test["site_id"].values, y, pred)
            boot_store[(cond, L)] = c
            s = scores(c.sum())
            rows.append({"condition": cond, "delay_h": L,
                         "recall": s["recall"], "precision": s["precision"],
                         "cost_weighted_loss": s["cost"],
                         "avg_precision": average_precision_score(y, prob),
                         "flagged_pct": 100 * pred.mean()})
            print(f"  delay {L:>3}h  recall {s['recall']:.3f}  precision {s['precision']:.3f}  "
                  f"AP {rows[-1]['avg_precision']:.3f}  cost {s['cost']:.4f}")

    res = pd.DataFrame(rows)

    # ---- paired bootstrap over stations ----------------------------------
    stations = boot_store[("monitored", 0)].index.values
    draws = rng.choice(len(stations), size=(args.n_boot, len(stations)), replace=True)
    ci_rows = []
    for cond in groups:
        base = boot_store[(cond, 0)].reindex(stations).fillna(0).values
        for L in lags:
            if L == 0:
                continue
            cur = boot_store[(cond, L)].reindex(stations).fillna(0).values
            d_rec, d_prec, d_cost = [], [], []
            for d in draws:
                b = dict(zip(["tp", "fp", "fn", "n"], base[d].sum(axis=0)))
                k = dict(zip(["tp", "fp", "fn", "n"], cur[d].sum(axis=0)))
                sb, sk = scores(b), scores(k)
                d_rec.append(sk["recall"] - sb["recall"])
                d_prec.append(sk["precision"] - sb["precision"])
                d_cost.append(sk["cost"] - sb["cost"])
            for name, arr in [("recall", d_rec), ("precision", d_prec), ("cost", d_cost)]:
                arr = np.array(arr, dtype=float)
                arr = arr[~np.isnan(arr)]
                lo, hi = np.percentile(arr, [2.5, 97.5])
                ci_rows.append({"condition": cond, "delay_h": L, "metric": name,
                                "change": float(np.mean(arr)),
                                "ci_low": float(lo), "ci_high": float(hi),
                                "excludes_zero": bool(lo > 0 or hi < 0)})
    ci = pd.DataFrame(ci_rows)

    # ---- report ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("ACCURACY BY DATA DELAY (test year 2024)")
    print("=" * 78)
    for cond in groups:
        print(f"\n{cond}")
        t = res[res.condition == cond][["delay_h", "recall", "precision",
                                        "avg_precision", "cost_weighted_loss",
                                        "flagged_pct"]]
        print(t.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("CHANGE FROM NO DELAY, 95% CI (paired bootstrap over stations)")
    print("=" * 78)
    for cond in groups:
        print(f"\n{cond}")
        for L in [l for l in lags if l > 0]:
            parts = []
            for m in ["recall", "precision"]:
                r = ci[(ci.condition == cond) & (ci.delay_h == L) & (ci.metric == m)].iloc[0]
                flag = "*" if r.excludes_zero else " "
                parts.append(f"{m} {r.change:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]{flag}")
            print(f"  {L:>3}h   " + "   ".join(parts))
    print("\n  * interval excludes zero")
    lo_p, hi_p = PRODUCTION_RANGE
    print(f"  The live system's inputs are roughly {lo_p}-{hi_p} hours stale; read the "
          f"rows that bracket that range.")

    RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_DIR / "data_delay.csv", index=False)
    ci.to_csv(RESULTS_DIR / "data_delay_ci.csv", index=False)
    (RESULTS_DIR / "data_delay_summary.json").write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "design": "train 2023, test 2024; station-derived inputs delayed by L hours",
        "model": "LightGBM" if HAVE_LGBM else "HistGradientBoosting",
        "thresholds": THRESHOLDS,
        "lags_h": lags,
        "test_rows": int(len(test)),
        "test_exceedances": int(test["label"].sum()),
        "production_staleness_h": list(PRODUCTION_RANGE),
        "n_bootstrap": args.n_boot,
    }, indent=2))
    print(f"\nSaved to {RESULTS_DIR}/data_delay.csv, data_delay_ci.csv, data_delay_summary.json")


if __name__ == "__main__":
    main()