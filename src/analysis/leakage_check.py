"""
Does sharing days between training and testing inflate the validation?

Card: AIR-20 — Spatiotemporal leakage check

THE CONCERN
    The Assessment 2 validation held out one station at a time, but trained
    on the other 17 stations over the same two years it tested on. Smoke
    events and inversions affect the whole basin at once, so on any given
    day the neighbours' outcomes 24 hours ahead closely match the held-out
    station's. A model can learn those day-specific outcomes from the
    neighbours, which a real forecast never has, because no station's future
    is known when the forecast is made.

    A train-2023, test-2024 run (data_delay.py) gave far lower skill than the
    leave-one-station-out validation, which is what this check investigates.

DESIGN
    For each of the 18 stations, test on that station's 2024 data and train
    on the other 17 stations under two setups of equal size:

        same period     other stations, 2024   (training shares the test days)
        earlier period  other stations, 2023   (no shared days)

    Test rows and training size are matched, so a difference between the two
    comes from sharing days, not from more data. Persistence and predicting
    nothing are scored on the same rows for reference.

    Per-station metrics use the A2 decision threshold, tau = 1/(1 + 10). A
    pooled threshold sweep gives the best achievable cost for each setup.
    Differences carry 95% intervals from a paired bootstrap over stations.

READING THE RESULT
    If skill drops sharply from same period to earlier period, the A2
    leave-one-station-out figures are optimistic and a validation that also
    separates time is needed. If the two are close, the A2 figures stand.

Usage:
    python3 src/analysis/leakage_check.py
    python3 src/analysis/leakage_check.py --conditions unmonitored
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

COST_RATIO = 10.0
TAU = 1.0 / (1.0 + COST_RATIO)
SETUPS = {"same_period": 2024, "earlier_period": 2023}
TEST_YEAR = 2024


def feature_sets(df):
    spatial = [c for c in df.columns if c.startswith(SPATIAL_PREFIX)]
    own = [c for c in df.columns
           if c.startswith(OWN_SENSOR_PREFIXES) or c in OWN_SENSOR_EXACT]
    fixed = ([c for c in WEATHER if c in df.columns]
             + [c for c in CALENDAR if c in df.columns]
             + [c for c in GEO if c in df.columns])
    return {"monitored": spatial + fixed + own, "unmonitored": spatial + fixed}


def make_model():
    """Same configuration as the validated model."""
    if HAVE_LGBM:
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=42)
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=50, random_state=42)


def counts(y, pred):
    return {"tp": int(((y == 1) & (pred == 1)).sum()),
            "fp": int(((y == 0) & (pred == 1)).sum()),
            "fn": int(((y == 1) & (pred == 0)).sum()),
            "n": int(len(y))}


def rp(c):
    rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else np.nan
    prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else np.nan
    return rec, prec


def best_cost(y, prob):
    best = np.inf
    for t in np.arange(0.005, 0.505, 0.005):
        pred = prob >= t
        fn = int(((y == 1) & ~pred).sum())
        fp = int(((y == 0) & pred).sum())
        best = min(best, (COST_RATIO * fn + fp) / len(y))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conditions", nargs="+", default=["monitored", "unmonitored"],
                    choices=["monitored", "unmonitored"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("Loading features...")
    df = pd.read_parquet(FEATURES_DIR).dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["year"] = pd.to_datetime(df["timestamp"]).dt.year
    feats = feature_sets(df)
    sites = sorted(df["site_id"].unique())
    print(f"  {len(df):,} rows, {len(sites)} stations; test year {TEST_YEAR}")
    print(f"  decision threshold {TAU:.3f}; "
          f"{len(sites)} folds x {len(SETUPS)} setups x {len(args.conditions)} conditions")

    fold_rows = []
    pooled = {(s, c): ([], []) for s in SETUPS for c in args.conditions}
    station_counts = {(s, c): {} for s in SETUPS for c in args.conditions}
    persist_counts, persist_y, persist_pred = {}, [], []

    for i, sid in enumerate(sites, 1):
        test = df[(df["site_id"] == sid) & (df["year"] == TEST_YEAR)]
        if test.empty or test["label"].sum() == 0:
            continue
        y = test["label"].values
        name = test["site_name"].iloc[0]

        if "PM2.5" in test.columns:
            pp = (test["PM2.5"].fillna(0).clip(lower=0) > 25).astype(int).values
            persist_counts[sid] = counts(y, pp)
            persist_y.append(y); persist_pred.append(pp)

        line = []
        for setup, train_year in SETUPS.items():
            train = df[(df["site_id"] != sid) & (df["year"] == train_year)]
            for cond in args.conditions:
                model = make_model().fit(train[feats[cond]], train["label"].values)
                prob = model.predict_proba(test[feats[cond]])[:, 1]
                pred = (prob >= TAU).astype(int)
                c = counts(y, pred)
                station_counts[(setup, cond)][sid] = c
                pooled[(setup, cond)][0].append(y)
                pooled[(setup, cond)][1].append(prob)
                rec, prec = rp(c)
                fold_rows.append({"site_id": sid, "site_name": name, "setup": setup,
                                  "condition": cond, "recall": rec, "precision": prec,
                                  "events": int(y.sum()), "train_rows": len(train), **c})
                if cond == args.conditions[-1]:
                    line.append(f"{setup.replace('_period', '')} {rec:.3f}")
        print(f"  {i:>2}/{len(sites)}  {name:<18} {int(y.sum()):>4} events   "
              f"recall ({args.conditions[-1]}): " + "  ".join(line))

    folds = pd.DataFrame(fold_rows)

    # ---- summary ------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"SAME PERIOD vs EARLIER PERIOD  (test: each station's {TEST_YEAR} data)")
    print("=" * 78)
    summary_rows = []
    for cond in args.conditions:
        for setup in SETUPS:
            f = folds[(folds.setup == setup) & (folds.condition == cond)]
            y = np.concatenate(pooled[(setup, cond)][0])
            p = np.concatenate(pooled[(setup, cond)][1])
            tot = {k: int(f[k].sum()) for k in ["tp", "fp", "fn", "n"]}
            prec_pooled = rp(tot)[1]
            summary_rows.append({
                "condition": cond, "setup": setup,
                "mean_recall": f["recall"].mean(), "mean_precision": f["precision"].mean(),
                "pooled_precision": prec_pooled,
                "avg_precision": average_precision_score(y, p),
                "best_cost": best_cost(y, p),
            })
    summ = pd.DataFrame(summary_rows)

    y_all = np.concatenate(pooled[(list(SETUPS)[0], args.conditions[0])][0])
    nothing_cost = COST_RATIO * y_all.mean()
    print(summ.round(4).to_string(index=False))
    print(f"\nReference on the same test rows ({len(y_all):,} rows, "
          f"{int(y_all.sum()):,} events, {y_all.mean():.2%} prevalence)")
    print(f"  predict nothing   cost {nothing_cost:.4f}")
    if persist_y:
        py, pp = np.concatenate(persist_y), np.concatenate(persist_pred)
        pr, ppr = rp(counts(py, pp))
        pc = (COST_RATIO * ((py == 1) & (pp == 0)).sum() + ((py == 0) & (pp == 1)).sum()) / len(py)
        print(f"  persistence       recall {pr:.3f}  precision {ppr:.3f}  cost {pc:.4f}")

    # ---- paired bootstrap over stations -----------------------------------
    print("\n" + "=" * 78)
    print("CHANGE FROM SAME PERIOD TO EARLIER PERIOD, 95% CI (paired over stations)")
    print("=" * 78)
    ci_rows = []
    for cond in args.conditions:
        a = station_counts[("same_period", cond)]
        b = station_counts[("earlier_period", cond)]
        keys = [k for k in a if k in b]
        A = np.array([[a[k][m] for m in ["tp", "fp", "fn"]] for k in keys], float)
        B = np.array([[b[k][m] for m in ["tp", "fp", "fn"]] for k in keys], float)
        draws = rng.choice(len(keys), size=(args.n_boot, len(keys)), replace=True)
        d_rec, d_prec = [], []
        for d in draws:
            ta, fa, na = A[d].sum(0)
            tb, fb, nb = B[d].sum(0)
            ra = ta / (ta + na) if ta + na else np.nan
            rb = tb / (tb + nb) if tb + nb else np.nan
            pa = ta / (ta + fa) if ta + fa else np.nan
            pb = tb / (tb + fb) if tb + fb else np.nan
            d_rec.append(rb - ra); d_prec.append(pb - pa)
        for metric, arr in [("recall", d_rec), ("precision", d_prec)]:
            arr = np.array(arr, float); arr = arr[~np.isnan(arr)]
            lo, hi = np.percentile(arr, [2.5, 97.5])
            ci_rows.append({"condition": cond, "metric": metric,
                            "change": float(arr.mean()), "ci_low": float(lo),
                            "ci_high": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)})
            print(f"  {cond:<12} {metric:<10} {arr.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]"
                  f"{'  *' if (lo > 0 or hi < 0) else ''}")
    print("\n  * interval excludes zero. A large negative change means sharing days "
          "inflates the validation.")

    RESULTS_DIR.mkdir(exist_ok=True)
    folds.to_csv(RESULTS_DIR / "leakage_check_folds.csv", index=False)
    summ.to_csv(RESULTS_DIR / "leakage_check_summary.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(RESULTS_DIR / "leakage_check_ci.csv", index=False)
    (RESULTS_DIR / "leakage_check.json").write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "design": "hold out each station; test its 2024 data; train other stations "
                  "in 2024 (same period) or 2023 (earlier period)",
        "model": "LightGBM" if HAVE_LGBM else "HistGradientBoosting",
        "decision_threshold": TAU,
        "predict_nothing_cost": nothing_cost,
        "n_bootstrap": args.n_boot,
    }, indent=2))
    print(f"\nSaved to {RESULTS_DIR}/leakage_check_*.csv and leakage_check.json")


if __name__ == "__main__":
    main()