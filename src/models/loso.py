
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except (ImportError, OSError):
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_LGBM = False

FEATURES_DIR = Path("data/processed/features_spatial")
RESULTS_DIR = Path("results")

CONDITIONS = ["monitored", "unmonitored", "weather_only"]

# Features derived from the target station's own sensor. Unavailable at an
# unmonitored location, so excluded from that condition.
OWN_SENSOR_PREFIXES = ("pm25_lag_", "pm25_mean_", "pm25_max_", "pm25_std_",
                       "pm25_delta_")
OWN_SENSOR_EXACT = {"PM2.5", "target_pm25_future", "imputed"}

# Features available anywhere: neighbours, weather, calendar, geography.
SPATIAL_PREFIX = "spatial_"
WEATHER = ["TEMP", "HUMID", "WSP", "WDR", "wind_u", "wind_v",
           "temp_lag_24h", "humid_lag_24h", "wsp_lag_24h", "wdr_lag_24h"]
CALENDAR = ["hour", "dayofweek", "month", "is_weekend",
            "hour_sin", "hour_cos", "month_sin", "month_cos"]
GEO = ["latitude", "longitude"]

def feature_sets(df):
    """Split columns into the three experimental conditions."""
    spatial = [c for c in df.columns if c.startswith(SPATIAL_PREFIX)]
    own = [c for c in df.columns
           if c.startswith(OWN_SENSOR_PREFIXES) or c in OWN_SENSOR_EXACT]
    weather = [c for c in WEATHER if c in df.columns]
    cal = [c for c in CALENDAR if c in df.columns]
    geo = [c for c in GEO if c in df.columns]

    weather_only = weather + cal + geo
    unmonitored = spatial + weather_only
    monitored = unmonitored + [c for c in own if c != "target_pm25_future"]
    return {"monitored": monitored,
            "unmonitored": unmonitored,
            "weather_only": weather_only}


def make_model():
    """
    Gradient-boosted trees, no class weighting.

    Imbalance is handled at the decision threshold instead. LightGBM is
    preferred but requires libomp, which is unavailable on some systems;
    scikit-learn's HistGradientBoosting is an equivalent fallback.
    """
    if HAVE_LGBM:
        return LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=42)
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=50, random_state=42)

def metrics(y_true, y_prob, threshold, cost_ratio):
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    n = len(y_true)

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "brier": round(float(np.mean((y_prob - y_true) ** 2)), 5),
        "cost_weighted_loss": round((cost_ratio * fn + fp) / n, 5),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_events": int(y_true.sum()), "n_rows": n,
    }


def fit_predict(df, held_out, feats):
    """Train on all stations except held_out; return probabilities for it."""
    train = df[df["site_id"] != held_out]
    test = df[df["site_id"] == held_out]

    ytr = train["label"].astype(int).values
    yte = test["label"].astype(int).values
    if ytr.sum() == 0 or yte.sum() == 0:
        return None, None

    model = make_model()
    model.fit(train[feats], ytr)
    return model.predict_proba(test[feats])[:, 1], yte

def threshold_sweep(prob, true, cost_ratio):
    """Sweep decision thresholds to find the cost-optimal operating point."""
    rows = []
    for t in np.arange(0.005, 0.505, 0.005):
        pred = (prob >= t).astype(int)
        tp = int(((true == 1) & (pred == 1)).sum())
        fp = int(((true == 0) & (pred == 1)).sum())
        fn = int(((true == 1) & (pred == 0)).sum())
        rows.append({
            "threshold": round(float(t), 3),
            "recall": round(tp / max(tp + fn, 1), 4),
            "precision": round(tp / max(tp + fp, 1), 4),
            "cost_weighted_loss": round((cost_ratio * fn + fp) / len(true), 5),
            "flagged_pct": round(float(pred.mean()) * 100, 2),
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cost-ratio", type=float, default=10.0,
                    help="Cost of a missed crossing relative to a false alarm")
    ap.add_argument("--decision-threshold", type=float, default=None,
                    help="Default: Bayes-optimal 1/(1+cost_ratio)")
    ap.add_argument("--no-sweep", action="store_true",
                    help="Skip the threshold sweep")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-fold probability diagnostics")
    args = ap.parse_args()

    tau = args.decision_threshold or 1.0 / (1.0 + args.cost_ratio)

    print("Loading spatial features...")
    df = pd.read_parquet(FEATURES_DIR)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    feats_by_cond = feature_sets(df)

    print(f"  {len(df):,} rows, {df['site_id'].nunique()} stations, "
          f"{df['label'].mean():.2%} positive")
    print(f"  monitored condition    : "
          f"{len(feats_by_cond['monitored'])} features (adds own-sensor history)")
    print(f"  unmonitored condition  : "
          f"{len(feats_by_cond['unmonitored'])} features "
          f"(spatial, weather, calendar, geography)")
    print(f"  weather-only condition : "
          f"{len(feats_by_cond['weather_only'])} features "
          f"(ablation — no spatial neighbours)")
    print(f"  model                  : "
          f"{'LightGBM' if HAVE_LGBM else 'HistGradientBoosting'}")
    print(f"  decision threshold     : {tau:.3f} "
          f"(cost ratio {args.cost_ratio}:1)")

    sites = sorted(df["site_id"].unique())
    rows = []
    pooled = {c: ([], []) for c in CONDITIONS}

    print(f"\nRunning {len(sites)} folds x {len(CONDITIONS)} conditions...")
    for i, sid in enumerate(sites, 1):
        info = df.loc[df["site_id"] == sid].iloc[0]
        name, region = info["site_name"], info["region"]
        nearest = float(info["spatial_nearest_dist"])

        fold = {}
        for cond in CONDITIONS:
            prob, yte = fit_predict(df, sid, feats_by_cond[cond])
            if prob is None:
                continue
            pooled[cond][0].append(prob)
            pooled[cond][1].append(yte)

            m = metrics(yte, prob, tau, args.cost_ratio)
            fold[cond] = m
            rows.append({"site_id": sid, "site_name": name, "region": region,
                         "nearest_station_km": round(nearest, 1),
                         "condition": cond, **m})

            if args.verbose:
                print(f"      {cond:<13} prob mean {prob.mean():.4f}, "
                      f"flagged {(prob >= tau).mean():.1%}")

        if {"monitored", "unmonitored"} <= set(fold):
            gap = fold["monitored"]["recall"] - fold["unmonitored"]["recall"]
            wo = fold.get("weather_only", {}).get("recall", float("nan"))
            print(f"  {i:>2}/{len(sites)}  {name:<20} "
                  f"recall {fold['monitored']['recall']:.3f} -> "
                  f"{fold['unmonitored']['recall']:.3f} -> {wo:.3f}  "
                  f"(sensor gap {gap:+.3f}, "
                  f"{fold['unmonitored']['n_events']} events)")

    res = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_DIR / "loso_results.csv", index=False)

    # ---- condition comparison --------------------------------------
    print("\n" + "=" * 78)
    print("CONDITION COMPARISON: monitored, unmonitored, weather-only")
    print("=" * 78)
    summary = (res.groupby("condition")[
        ["recall", "precision", "f1", "brier", "cost_weighted_loss"]]
        .mean().round(4).reindex(CONDITIONS))
    print(summary.to_string())

    mon = res[res.condition == "monitored"]["recall"].mean()
    unm = res[res.condition == "unmonitored"]["recall"].mean()
    wth = res[res.condition == "weather_only"]["recall"].mean()

    print(f"\nMean recall with own sensor     : {mon:.3f}")
    print(f"Mean recall without own sensor  : {unm:.3f}")
    print(f"Mean recall without spatial     : {wth:.3f}")
    print(f"\nReliability penalty (no sensor)  : {(mon - unm) / mon:.1%} "
          f"of monitored recall")
    print(f"Spatial feature contribution     : {(unm - wth) / unm:.1%} "
          f"of unmonitored recall")



    # ---- per-station breakdown -------------------------------------
    piv = res.pivot_table(index=["site_name", "region", "nearest_station_km"],
                          columns="condition", values="recall").reset_index()
    piv["sensor_gap"] = piv["monitored"] - piv["unmonitored"]
    piv["spatial_gap"] = piv["unmonitored"] - piv["weather_only"]
    piv = piv.sort_values("sensor_gap", ascending=False)

    print("\n" + "=" * 78)
    print("PER-STATION RECALL BY CONDITION (largest sensor gap first)")
    print("=" * 78)
    print(piv.round(3).to_string(index=False))

    c = piv["sensor_gap"].corr(piv["nearest_station_km"])
    print(f"\ncorr(sensor gap, distance to nearest station) = {c:.3f}")
    print("  Weak — penalty is not simply a function of distance"
          if abs(c) < 0.3 else "  Isolated stations suffer a larger penalty")

    print("\nMean sensor gap by region")
    print(piv.groupby("region")["sensor_gap"].mean().round(3).to_string())
    piv.to_csv(RESULTS_DIR / "loso_reliability_gap.csv", index=False)

    # ---- threshold sweep -------------------------------------------
    best_by_cond = {}
    if not args.no_sweep:
        print("\n" + "=" * 78)
        print("THRESHOLD SWEEP (pooled across all folds)")
        print("=" * 78)
        print("Note: the optimum is selected on the same folds it is "
              "evaluated on, so it\nis mildly optimistic. In deployment the "
              "threshold would be fixed using\nout-of-fold validation on an "
              "earlier time partition.\n")

        baseline_cost = args.cost_ratio * df["label"].mean()
        print(f"Always-negative cost-weighted loss: {baseline_cost:.5f}\n")

        for cond in CONDITIONS:
            prob = np.concatenate(pooled[cond][0])
            true = np.concatenate(pooled[cond][1])
            sweep = threshold_sweep(prob, true, args.cost_ratio)
            best = sweep.loc[sweep["cost_weighted_loss"].idxmin()]
            best_by_cond[cond] = best

            print(f"{cond}")
            print(sweep.iloc[::10][["threshold", "recall", "precision",
                                    "cost_weighted_loss", "flagged_pct"]]
                  .to_string(index=False))
            print(f"  optimal threshold {best['threshold']}: "
                  f"recall {best['recall']:.3f}, "
                  f"precision {best['precision']:.3f}, "
                  f"cost {best['cost_weighted_loss']:.5f}")
            verdict = ("BEATS" if best["cost_weighted_loss"] < baseline_cost
                       else "does NOT beat")
            improvement = (1 - best["cost_weighted_loss"] / baseline_cost) * 100
            print(f"  model {verdict} predicting nothing "
                  f"({best['cost_weighted_loss']:.5f} vs {baseline_cost:.5f}, "
                  f"{improvement:+.0f}%)\n")

            sweep.to_csv(RESULTS_DIR / f"threshold_sweep_{cond}.csv",
                         index=False)

    # ---- lineage ----------------------------------------------------
    with open(RESULTS_DIR / "loso_config.json", "w") as f:
        json.dump({
            "model": "LightGBM" if HAVE_LGBM else "HistGradientBoosting",
            "class_weighting": "none — imbalance handled at decision threshold",
            "n_folds": len(sites),
            "conditions": CONDITIONS,
            "decision_threshold": tau,
            "cost_ratio": args.cost_ratio,
            "n_features": {c: len(f) for c, f in feats_by_cond.items()},
            "mean_recall": {"monitored": float(mon),
                            "unmonitored": float(unm),
                            "weather_only": float(wth)},
            "reliability_penalty_no_sensor": float((mon - unm) / mon),
            "spatial_feature_contribution": float((unm - wth) / unm),
            "optimal_thresholds": {
                c: float(b["threshold"]) for c, b in best_by_cond.items()},
            "threshold_selection_caveat": (
                "optimum selected in-fold; mildly optimistic"),
        }, f, indent=2)

    print(f"Saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()