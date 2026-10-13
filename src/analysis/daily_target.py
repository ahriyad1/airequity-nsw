"""
Can we forecast tomorrow's daily-average PM2.5 exceedance on unseen days?

==Daily target under an honest validation

WHY A DAILY TARGET
    The national PM2.5 standard (25 ug/m3) is a 24-hour average, but the A2
    model predicted single hours crossing 25 a day ahead. Hourly readings
    spike and fall unpredictably; the daily average is smoother and is what
    an air-quality advisory is actually about. This script asks: is a station
    over the 24-hour standard tomorrow?
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

OBS_DIR = Path("data/processed/observations")
RESULTS_DIR = Path("results")

THRESHOLD = 25.0          # ug/m3, national 24-hour PM2.5 standard
MIN_HOURS = 18            # valid hours needed for a daily mean
COST_RATIO = 10.0
TAU = 1.0 / (1.0 + COST_RATIO)
EVENING = range(18, 24)   # hours used for the evening mean


# ----- data ----

def parse_years(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(y) for y in s.split(",")]


def load_daily():
    #ourly observations -> one row per station-day
    obs = pd.read_parquet(OBS_DIR)
    obs = obs[obs["frequency"] == "Hourly average"]
    sites = (obs[["site_id", "site_name", "latitude", "longitude"]]
             .drop_duplicates("site_id").set_index("site_id"))

    wide = obs.pivot_table(index=["site_id", "timestamp"], columns="parameter",
                           values="value", aggfunc="first").reset_index()
    wide["PM2.5"] = wide["PM2.5"].clip(lower=0)
    wide["day"] = wide["timestamp"].dt.floor("D")
    wide["hr"] = wide["timestamp"].dt.hour
    if "WSP" in wide and "WDR" in wide:
        rad = np.deg2rad(wide["WDR"])
        wide["wind_u"] = -wide["WSP"] * np.sin(rad)
        wide["wind_v"] = -wide["WSP"] * np.cos(rad)

    g = wide.groupby(["site_id", "day"])
    daily = g["PM2.5"].agg(pm_mean="mean", pm_max="max", pm_min="min",
                           pm_std="std", pm_n="count")
    eve = wide[wide["hr"].isin(EVENING)].groupby(["site_id", "day"])["PM2.5"].mean()
    daily["pm_eve"] = eve
    for c in ["TEMP", "HUMID", "WSP", "wind_u", "wind_v"]:
        if c in wide:
            daily[f"w_{c.lower()}"] = g[c].mean()
    daily = daily.reset_index()

    bad = daily["pm_n"] < MIN_HOURS
    daily.loc[bad, ["pm_mean", "pm_max", "pm_min", "pm_std", "pm_eve"]] = np.nan
    return daily, sites


def complete_grid(daily):
    """Every station gets every day between its first and last record."""
    frames = []
    for sid, g in daily.groupby("site_id"):
        days = pd.date_range(g["day"].min(), g["day"].max(), freq="D")
        frames.append(g.set_index("day").reindex(days).rename_axis("day")
                      .assign(site_id=sid).reset_index())
    return pd.concat(frames, ignore_index=True)


def network_features(daily, sites):
    #Self-excluded network summaries of each day's station means
    pm = daily.pivot(index="day", columns="site_id", values="pm_mean")
    ids = list(pm.columns)
    lat = np.deg2rad(sites.loc[ids, "latitude"].values.astype(float))
    lon = np.deg2rad(sites.loc[ids, "longitude"].values.astype(float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    dist = 2 * 6371 * np.arcsin(np.sqrt(a))
    np.fill_diagonal(dist, np.inf)
    w = 1.0 / dist ** 2                      # zero weight on self

    X = pm.values                            # days x stations
    valid = ~np.isnan(X)
    X0 = np.where(valid, X, 0.0)
    out = []
    for j, sid in enumerate(ids):
        others = np.ones(len(ids), bool); others[j] = False
        Xo, Vo = X[:, others], valid[:, others]
        wo = w[j, others]
        wsum = (Vo * wo).sum(1)
        idw = np.where(wsum > 0, (X0[:, others] * wo).sum(1) / np.where(wsum > 0, wsum, 1), np.nan)
        # nearest station that reported that day
        order = np.argsort(dist[j, others])
        Xs = Xo[:, order]
        first = np.argmax(~np.isnan(Xs), axis=1)
        nn = Xs[np.arange(len(Xs)), first]
        with np.errstate(all="ignore"):
            out.append(pd.DataFrame({
                "day": pm.index, "site_id": sid,
                "net_mean": np.nanmean(Xo, 1), "net_max": np.nanmax(Xo, 1),
                "net_n_over": (Xo > THRESHOLD).sum(1),
                "net_idw": idw, "net_nearest": nn,
            }))
    return pd.concat(out, ignore_index=True)


def build_table(daily, sites, delay):
    #One row per (station, target day T) with features from base day B
    d = complete_grid(daily).merge(network_features(daily, sites),
                                   on=["day", "site_id"], how="left")
    d = d.sort_values(["site_id", "day"]).reset_index(drop=True)
    g = d.groupby("site_id")

    feat = pd.DataFrame({"site_id": d["site_id"], "base_day": d["day"]})
    own_cols = ["pm_mean", "pm_max", "pm_min", "pm_std", "pm_eve"]
    net_cols = ["net_mean", "net_max", "net_n_over", "net_idw", "net_nearest"]
    w_cols = [c for c in d.columns if c.startswith("w_")]
    for c in own_cols + net_cols + w_cols:
        feat[c] = d[c]
    feat["pm_mean_lag1"] = g["pm_mean"].shift(1)
    feat["pm_mean_lag2"] = g["pm_mean"].shift(2)
    feat["pm_mean_3d"] = g["pm_mean"].transform(lambda s: s.rolling(3, min_periods=2).mean())
    feat["net_mean_lag1"] = g["net_mean"].shift(1)
    feat["net_idw_lag1"] = g["net_idw"].shift(1)

    # target day T = base day + 1 + delay, same station
    lead = 1 + delay
    feat["day"] = feat["base_day"] + pd.Timedelta(days=lead)
    tgt = d[["site_id", "day", "pm_mean"]].rename(columns={"pm_mean": "target"})
    t = feat.merge(tgt, on=["site_id", "day"], how="inner").dropna(subset=["target"])
    t["y"] = (t["target"] > THRESHOLD).astype(int)

    doy = t["day"].dt.dayofyear
    t["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    t["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    t["month"] = t["day"].dt.month
    t["dayofweek"] = t["day"].dt.dayofweek
    t["year"] = t["day"].dt.year

    own = own_cols + ["pm_mean_lag1", "pm_mean_lag2", "pm_mean_3d"]
    net = net_cols + ["net_mean_lag1", "net_idw_lag1"]
    cal = ["doy_sin", "doy_cos", "month", "dayofweek"]
    fsets = {"with_sensor": own + net + w_cols + cal,
             "without_sensor": net + w_cols + cal}
    return t, fsets


# ---- metrics ----

def make_model():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, random_state=42)


def tally(y, alarm):
    y = np.asarray(y).astype(bool); a = np.asarray(alarm).astype(bool)
    return {"n": len(y), "events": int(y.sum()), "caught": int((y & a).sum()),
            "false_alarms": int((~y & a).sum()), "missed": int((y & ~a).sum())}


def describe(c):
    rec = c["caught"] / c["events"] if c["events"] else np.nan
    al = c["caught"] + c["false_alarms"]
    prec = c["caught"] / al if al else np.nan
    cost = (COST_RATIO * c["missed"] + c["false_alarms"]) / c["n"] if c["n"] else np.nan
    return rec, prec, cost


def boot_diff(per_station, a, b, rng, n_boot):
   #Bootstrap over stations: cost(a) - cost(b)
    keys = sorted(per_station)
    arr = {m: np.array([[per_station[k][m][f] for f in ("n", "missed", "false_alarms")]
                        for k in keys], float) for m in (a, b)}
    draws = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    diffs = []
    for d in draws:
        ca = arr[a][d].sum(0); cb = arr[b][d].sum(0)
        diffs.append((COST_RATIO * ca[1] + ca[2]) / ca[0] - (COST_RATIO * cb[1] + cb[2]) / cb[0])
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


# ---- main ----

def run(table, fsets, train_years, test_years, rng, n_boot, delay):
    sites = sorted(table.loc[table["year"].isin(test_years), "site_id"].unique())
    per_station, probs, rows = {}, {k: ([], []) for k in fsets}, []

    for sid in sites:
        test = table[(table["site_id"] == sid) & table["year"].isin(test_years)]
        train = table[(table["site_id"] != sid) & table["year"].isin(train_years)]
        if test.empty or train["y"].sum() == 0:
            continue
        y = test["y"].values
        res = {}
        alarms = {}
        for cond, cols in fsets.items():
            p = make_model().fit(train[cols], train["y"]).predict_proba(test[cols])[:, 1]
            probs[cond][0].append(y); probs[cond][1].append(p)
            alarms[cond] = p >= TAU
        alarms["persistence"] = (test["pm_mean"] > THRESHOLD).values
        alarms["network_persistence"] = (test["net_idw"] > THRESHOLD).values
        alarms["never"] = np.zeros(len(y), bool)
        alarms["with_sensor_or_persistence"] = alarms["with_sensor"] | alarms["persistence"]
        for k, a in alarms.items():
            res[k] = tally(y, a)
            rows.append({"delay_days": delay, "site_id": sid, "method": k,
                         "year_split": "all", **res[k]})
            for yr in test_years:
                m = (test["year"] == yr).values
                if m.any():
                    rows.append({"delay_days": delay, "site_id": sid, "method": k,
                                 "year_split": yr, **tally(y[m], a[m])})
        per_station[sid] = res

    folds = pd.DataFrame(rows)
    summary = []
    for (split, method), f in folds.groupby(["year_split", "method"], sort=False):
        tot = {c: int(f[c].sum()) for c in ["n", "events", "caught", "false_alarms", "missed"]}
        rec, prec, cost = describe(tot)
        row = {"delay_days": delay, "test_years": str(split), "method": method, **tot,
               "recall": rec, "precision": prec, "cost": cost}
        if method in probs and split == "all" and probs[method][0]:
            yy = np.concatenate(probs[method][0]); pp = np.concatenate(probs[method][1])
            row["avg_precision"] = average_precision_score(yy, pp) if yy.sum() else np.nan
        summary.append(row)
    summary = pd.DataFrame(summary)

    comparisons = [("with_sensor", "persistence"),
                   ("without_sensor", "network_persistence"),
                   ("with_sensor", "without_sensor"),
                   ("with_sensor_or_persistence", "persistence"),
                   ("with_sensor", "never"), ("without_sensor", "never")]
    ci = []
    for a, b in comparisons:
        mean, lo, hi = boot_diff(per_station, a, b, rng, n_boot)
        ci.append({"delay_days": delay, "comparison": f"{a} - {b}", "cost_diff": mean,
                   "ci_low": lo, "ci_high": hi, "excludes_zero": bool(lo > 0 or hi < 0)})
    return folds, summary, pd.DataFrame(ci)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-years", default="2023")
    ap.add_argument("--test-years", default="2024")
    ap.add_argument("--delays", nargs="+", type=int, default=[0])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train_years, test_years = parse_years(args.train_years), parse_years(args.test_years)
    if set(train_years) & set(test_years):
        ap.error("training and test years must not overlap")
    rng = np.random.default_rng(args.seed)

    print("Loading observations...")
    daily, sites = load_daily()
    print(f"  {len(daily):,} station-days, {daily['site_id'].nunique()} stations, "
          f"{daily['day'].min().date()} to {daily['day'].max().date()}")

    all_folds, all_summ, all_ci = [], [], []
    for delay in args.delays:
        table, fsets = build_table(daily, sites, delay)
        tr = table[table["year"].isin(train_years)]
        te = table[table["year"].isin(test_years)]
        print(f"\n=== delay {delay} day(s): features from day T-{1 + delay} ===")
        print(f"  train {train_years[0]}-{train_years[-1]}: {len(tr):,} station-days, "
              f"{int(tr['y'].sum())} over 25 on {tr.loc[tr['y'] == 1, 'day'].nunique()} dates")
        print(f"  test  {test_years[0]}-{test_years[-1]}: {len(te):,} station-days, "
              f"{int(te['y'].sum())} over 25 on {te.loc[te['y'] == 1, 'day'].nunique()} dates")
        folds, summ, ci = run(table, fsets, train_years, test_years, rng, args.n_boot, delay)
        all_folds.append(folds); all_summ.append(summ); all_ci.append(ci)

        show = summ[summ["test_years"] == "all"]
        print("\n  " + show[["method", "events", "caught", "false_alarms", "recall",
                             "precision", "cost"]].round(4).to_string(index=False)
              .replace("\n", "\n  "))
        print("\n  Cost difference, 95% CI over stations (negative = first is better):")
        for r in ci.itertuples():
            print(f"    {r.comparison:<45} {r.cost_diff:+.4f}  [{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
                  f"{'  *' if r.excludes_zero else ''}")

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.concat(all_folds).to_csv(RESULTS_DIR / "daily_target_folds.csv", index=False)
    pd.concat(all_summ).to_csv(RESULTS_DIR / "daily_target_summary.csv", index=False)
    pd.concat(all_ci).to_csv(RESULTS_DIR / "daily_target_ci.csv", index=False)
    (RESULTS_DIR / "daily_target.json").write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "train_years": train_years, "test_years": test_years, "delays": args.delays,
        "threshold": THRESHOLD, "min_valid_hours": MIN_HOURS,
        "cost_ratio": COST_RATIO, "decision_threshold": TAU,
        "validation": "hold out each station AND the test years; train on other "
                      "stations in the training years only",
        "n_bootstrap": args.n_boot,
    }, indent=2))
    print(f"\nSaved to {RESULTS_DIR}/daily_target_*.csv and daily_target.json")


if __name__ == "__main__":
    main()