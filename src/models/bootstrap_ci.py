"""
Bootstrap confidence intervals for the LOSO effect sizes.

Card: AIR-16 — Uncertainty quantification

WHY THIS EXISTS
    LOSO produces 18 paired observations — one per held-out station. Mean
    differences computed from 18 folds, each containing between 78 and 406
    events, carry real sampling uncertainty. Reporting "2.9%" as a bare
    point estimate invites the obvious question: is that distinguishable
    from zero?

METHOD
    Paired bootstrap over stations. Stations are resampled with
    replacement; within each resample the paired condition results stay
    together, preserving the pairing that makes the comparison meaningful.
    10,000 resamples, percentile interval.

    A paired sign test is also reported as a distribution-free check on
    the direction of the effect.

WHAT IS BEING TESTED
    sensor_gap    monitored recall minus unmonitored recall.
                  The penalty for having no local sensor.

    spatial_gap   unmonitored recall minus weather-only recall.
                  The contribution of neighbouring-station information.

Usage:
    python3 src/models/bootstrap_ci.py
    python3 src/models/bootstrap_ci.py --n-boot 20000 --metric precision
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
LOSO_RESULTS = RESULTS_DIR / "loso_results.csv"


def paired_bootstrap(values, n_boot, rng):
    """
    Resample stations with replacement; return bootstrap means.

    Resampling at station level rather than row level is what makes this
    paired: each draw takes a station's result under both conditions
    together, so the difference is never computed across mismatched sites.
    """
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    return values[idx].mean(axis=1)


def sign_test(diffs):
    """
    Distribution-free check on effect direction.

    Counts how many stations show a positive difference and returns the
    two-sided binomial p-value under the null that positive and negative
    are equally likely.
    """
    from math import comb
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    k = int((nonzero > 0).sum())
    if n == 0:
        return k, n, 1.0
    tail = min(k, n - k)
    p = 2 * sum(comb(n, i) for i in range(tail + 1)) / (2 ** n)
    return k, n, min(p, 1.0)


def summarise(name, diffs, base, n_boot, rng, alpha=0.05):
    """Bootstrap CI, relative effect, and sign test for one comparison."""
    obs = float(diffs.mean())
    boot = paired_bootstrap(diffs.values, n_boot, rng)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    rel = obs / base if base else float("nan")
    rel_lo, rel_hi = (lo / base, hi / base) if base else (np.nan, np.nan)

    k, n, p = sign_test(diffs)
    crosses_zero = lo <= 0 <= hi

    print(f"\n{name}")
    print(f"  observed difference : {obs:+.4f}")
    print(f"  95% CI              : [{lo:+.4f}, {hi:+.4f}]")
    print(f"  relative to base    : {rel:+.1%} "
          f"(CI {rel_lo:+.1%} to {rel_hi:+.1%})")
    print(f"  sign test           : {k}/{n} stations positive, p = {p:.3f}")
    print(f"  interpretation      : " + (
        "CI includes zero — effect not distinguishable from no difference"
        if crosses_zero else
        "CI excludes zero — effect is consistent across stations"))

    return {
        "observed": round(obs, 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "relative": round(rel, 4),
        "relative_ci_low": round(float(rel_lo), 4),
        "relative_ci_high": round(float(rel_hi), 4),
        "stations_positive": k,
        "stations_nonzero": n,
        "sign_test_p": round(p, 4),
        "ci_excludes_zero": bool(not crosses_zero),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=10000,
                    help="Bootstrap resamples (default 10,000)")
    ap.add_argument("--metric", default="recall",
                    choices=["recall", "precision", "f1",
                             "cost_weighted_loss"],
                    help="Metric to test (default recall)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not LOSO_RESULTS.exists():
        print(f"{LOSO_RESULTS} not found — run loso_validation.py first.")
        return

    rng = np.random.default_rng(args.seed)
    res = pd.read_csv(LOSO_RESULTS)

    wide = res.pivot_table(index=["site_id", "site_name", "region"],
                           columns="condition", values=args.metric)
    wide = wide.dropna(subset=["monitored", "unmonitored", "weather_only"])

    print("=" * 78)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS — metric: {args.metric}")
    print("=" * 78)
    print(f"  paired resampling over {len(wide)} stations, "
          f"{args.n_boot:,} resamples")
    print(f"\nMean {args.metric} by condition")
    for c in ["monitored", "unmonitored", "weather_only"]:
        print(f"  {c:<14} {wide[c].mean():.4f}")

    sensor = wide["monitored"] - wide["unmonitored"]
    spatial = wide["unmonitored"] - wide["weather_only"]

    out = {
        "metric": args.metric,
        "n_stations": int(len(wide)),
        "n_bootstrap": args.n_boot,
        "mean_by_condition": {
            c: round(float(wide[c].mean()), 4)
            for c in ["monitored", "unmonitored", "weather_only"]},
    }

    out["sensor_gap"] = summarise(
        "SENSOR GAP  (monitored - unmonitored)\n"
        "  the penalty for having no local sensor",
        sensor, float(wide["monitored"].mean()), args.n_boot, rng)

    out["spatial_gap"] = summarise(
        "SPATIAL GAP  (unmonitored - weather_only)\n"
        "  the contribution of neighbouring-station information",
        spatial, float(wide["unmonitored"].mean()), args.n_boot, rng)

    # ---- which effect is larger? -----------------------------------
    diff_of_diffs = spatial - sensor
    boot = paired_bootstrap(diff_of_diffs.values, args.n_boot, rng)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nSPATIAL GAP minus SENSOR GAP")
    print(f"  observed            : {diff_of_diffs.mean():+.4f}")
    print(f"  95% CI              : [{lo:+.4f}, {hi:+.4f}]")
    print(f"  interpretation      : " + (
        "cannot conclude one effect exceeds the other"
        if lo <= 0 <= hi else
        "neighbour information contributes significantly more than "
        "own-sensor history"))
    out["spatial_minus_sensor"] = {
        "observed": round(float(diff_of_diffs.mean()), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "ci_excludes_zero": bool(not (lo <= 0 <= hi)),
    }

    # ---- per-station detail ----------------------------------------
    detail = wide.copy()
    detail["sensor_gap"] = sensor
    detail["spatial_gap"] = spatial
    detail = detail.reset_index().sort_values("spatial_gap", ascending=False)

    print("\n" + "=" * 78)
    print("PER-STATION EFFECTS (largest spatial contribution first)")
    print("=" * 78)
    print(detail[["site_name", "region", "monitored", "unmonitored",
                  "weather_only", "sensor_gap", "spatial_gap"]]
          .round(3).to_string(index=False))

    detail.to_csv(RESULTS_DIR / f"bootstrap_detail_{args.metric}.csv",
                  index=False)
    with open(RESULTS_DIR / f"bootstrap_ci_{args.metric}.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR}/bootstrap_ci_{args.metric}.json")


if __name__ == "__main__":
    main()