"""
Can we forecast tomorrow's daily-average PM2.5 exceedance on unseen days?

Daily target under an honest validation

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
from scipy.optimize import minimize
from scipy.stats import binomtest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

OBS_DIR = Path("data/processed/observations")
RESULTS_DIR = Path("results")

THRESHOLD = 25.0          # ug/m3, national 24-hour PM2.5 standard
MIN_HOURS = 18            # valid hours needed for a daily mean
COST_RATIO = 10.0
TAU = 1.0 / (1.0 + COST_RATIO)
EVENING = range(18, 24)   # hours used for the evening mean


# ----- data ----

def parse_years(s):
    #'2016-2018,2021' -> [2016, 2017, 2018, 2021]
    years = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            years += range(int(a), int(b) + 1)
        else:
            years.append(int(part))
    return sorted(set(years))


def load_daily(obs_dir=OBS_DIR):
    # Hourly observations -> one row per station-day.
    # Read one year at a time, only the columns and rows needed: the full
    # 2016-2025 store does not fit in memory with every column loaded.
    cols = ["site_id", "site_name", "latitude", "longitude",
            "parameter", "frequency", "timestamp", "value"]
    keep = ["PM2.5", "TEMP", "HUMID", "WSP", "WDR"]
    parts = []
    for year_dir in sorted(Path(obs_dir).glob("year=*")):
        part = pd.read_parquet(year_dir, columns=cols,
                               filters=[("frequency", "==", "Hourly average"),
                                        ("parameter", "in", keep)])
        part["timestamp"] = pd.to_datetime(part["timestamp"]).astype("datetime64[ns]")
        parts.append(part.drop(columns="frequency"))
    obs = pd.concat(parts, ignore_index=True)
    sites = (obs[["site_id", "site_name", "latitude", "longitude"]]
             .drop_duplicates("site_id").set_index("site_id"))

    wide = (obs.drop_duplicates(["site_id", "timestamp", "parameter"])
               .set_index(["site_id", "timestamp", "parameter"])["value"]
               .unstack("parameter").reset_index())
    wide.columns.name = None
    del obs, parts
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
   #Every station gets every day between its first and last record
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


OWN_COLS = ["pm_mean", "pm_max", "pm_min", "pm_std", "pm_eve"]
NET_COLS = ["net_mean", "net_max", "net_n_over", "net_idw", "net_nearest"]


def daily_features(daily, sites):
    # One row per (station, base day B): everything known at the end of B.
    # Shared by this experiment and the live forecast (predict_daily.py).
    d = complete_grid(daily).merge(network_features(daily, sites),
                                   on=["day", "site_id"], how="left")
    d = d.sort_values(["site_id", "day"]).reset_index(drop=True)
    g = d.groupby("site_id")

    feat = pd.DataFrame({"site_id": d["site_id"], "base_day": d["day"]})
    w_cols = [c for c in d.columns if c.startswith("w_")]
    for c in OWN_COLS + NET_COLS + w_cols:
        feat[c] = d[c]
    # A missing earlier day is filled with the newest day that is available,
    # and the 3-day mean uses whichever of the three days exist. Otherwise a
    # station that went offline during smoke (Oakdale, December 2019) gets its
    # gaps filled with a typical clean-air value and its forecast collapses.
    feat["pm_mean_lag1"] = g["pm_mean"].shift(1).fillna(feat["pm_mean"])
    feat["pm_mean_lag2"] = g["pm_mean"].shift(2).fillna(feat["pm_mean_lag1"])
    feat["pm_mean_3d"] = g["pm_mean"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    feat["net_mean_lag1"] = g["net_mean"].shift(1).fillna(feat["net_mean"])
    feat["net_idw_lag1"] = g["net_idw"].shift(1).fillna(feat["net_idw"])
    return feat, d, w_cols


def add_calendar(t):
    # Calendar of the target day T (known in advance)
    doy = t["day"].dt.dayofyear
    t["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    t["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    t["month"] = t["day"].dt.month
    t["dayofweek"] = t["day"].dt.dayofweek
    t["year"] = t["day"].dt.year
    return t


# Day-ahead weather forecast for the target day T (Open-Meteo Previous Runs,
# 24-hour lead), summarised per day. See src/ingest/fetch_weather_history.py.
WX_COLS = ["fx_wsp_mean", "fx_wsp_max", "fx_precip", "fx_rh_mean", "fx_temp_max",
           "fx_pres_mean", "fx_cloud_mean", "fx_u", "fx_v"]


def load_weather(path):
    """Hourly archived forecasts -> one row per (station, day)."""
    w = pd.read_parquet(path)
    w["day"] = pd.to_datetime(w["time"]).dt.floor("D").astype("datetime64[ns]")
    rad = np.deg2rad(w["wind_direction_10m"])
    w["u"] = -w["wind_speed_10m"] * np.sin(rad)
    w["v"] = -w["wind_speed_10m"] * np.cos(rad)
    g = w.groupby(["site_id", "day"])
    out = pd.DataFrame({
        "fx_wsp_mean": g["wind_speed_10m"].mean(),
        "fx_wsp_max": g["wind_speed_10m"].max(),
        "fx_precip": g["precipitation"].sum(min_count=1),
        "fx_rh_mean": g["relative_humidity_2m"].mean(),
        "fx_temp_max": g["temperature_2m"].max(),
        "fx_pres_mean": g["surface_pressure"].mean(),
        "fx_cloud_mean": g["cloud_cover"].mean(),
        "fx_u": g["u"].mean(),
        "fx_v": g["v"].mean(),
        "fx_hours": g["wind_speed_10m"].count(),
    }).reset_index()
    # a day needs most of its hours to count
    out.loc[out["fx_hours"] < 18, WX_COLS] = np.nan
    return out.drop(columns="fx_hours")


def feature_sets(w_cols):
    own = OWN_COLS + ["pm_mean_lag1", "pm_mean_lag2", "pm_mean_3d"]
    net = NET_COLS + ["net_mean_lag1", "net_idw_lag1"]
    cal = ["doy_sin", "doy_cos", "month", "dayofweek"]
    # Logistic model: pollution features only. The gradient-boosting model
    # also sees weather and calendar, and learned from the training years
    # that summer pollution usually clears by the next day; in the 2019-20
    # smoke it did not, and the model missed most bad days.
    pm_own = [c for c in own if c != "pm_std"]
    return {
        "logistic": {"with_sensor": pm_own + net, "without_sensor": net},
        # same, plus the day-ahead weather forecast for T (kept last: these
        # weights may take either sign)
        "logistic_wx": {"with_sensor": pm_own + net + WX_COLS,
                        "without_sensor": net + WX_COLS},
        "hgb": {"with_sensor": own + net + w_cols + cal,
                "without_sensor": net + w_cols + cal},
    }


def build_table(daily, sites, delay, weather=None):
    # One row per (station, target day T) with features from base day B.
    # With `weather`, only days that have a weather forecast are kept, for
    # every model, so models with and without weather see the same rows.
    feat, d, w_cols = daily_features(daily, sites)

    # target day T = base day + 1 + delay, same station
    lead = 1 + delay
    feat["day"] = feat["base_day"] + pd.Timedelta(days=lead)
    tgt = d[["site_id", "day", "pm_mean"]].rename(columns={"pm_mean": "target"})
    t = feat.merge(tgt, on=["site_id", "day"], how="inner").dropna(subset=["target"])
    t["y"] = (t["target"] > THRESHOLD).astype(int)
    if weather is not None:
        t = t.merge(weather, on=["site_id", "day"], how="inner")
        t = t.dropna(subset=WX_COLS, how="any")
    return add_calendar(t), feature_sets(w_cols)


# ---- metrics ----

def log_clip(X):
    return np.log1p(np.clip(X, 0, None))


def log_clip_first(X, n_free=0):
    """Log-transform the pollution columns; leave the last n_free as they are."""
    X = np.array(X, dtype=float)
    k = X.shape[1] - n_free
    X[:, :k] = np.log1p(np.clip(X[:, :k], 0, None))
    return X


class NonNegLogistic(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, n_free=0):
        self.C = C
        self.n_free = n_free   # last n_free weights may be negative (weather)

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        n, k = X.shape

        def loss(theta):
            w, b = theta[:k], theta[k]
            z = X @ w + b
            # log(1 + e^z) - y z, computed stably
            nll = np.sum(np.logaddexp(0, z) - y * z)
            p = 1.0 / (1.0 + np.exp(-z))
            g = p - y
            return (0.5 * w @ w + self.C * nll,
                    np.append(w + self.C * (X.T @ g), self.C * g.sum()))

        bounds = ([(0, None)] * (k - self.n_free) + [(None, None)] * self.n_free
                  + [(None, None)])
        res = minimize(loss, np.zeros(k + 1), jac=True, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 2000})
        self.coef_ = res.x[:k][None, :]
        self.intercept_ = res.x[k:]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        z = np.asarray(X, float) @ self.coef_[0] + self.intercept_[0]
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_model(kind):
    if kind == "logistic_wx":
        n = len(WX_COLS)
        return make_pipeline(SimpleImputer(strategy="median"),
                             FunctionTransformer(log_clip_first, kw_args={"n_free": n}),
                             StandardScaler(), NonNegLogistic(C=1.0, n_free=n))
    if kind == "logistic":
        # PM2.5 is heavily skewed, so features are log-transformed first.
        return make_pipeline(SimpleImputer(strategy="median"),
                             FunctionTransformer(log_clip), StandardScaler(),
                             NonNegLogistic(C=1.0))
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


def sign_test(per_station, a, b):
    #Stations where a has lower cost than b, vs higher (ties dropped)
    def cost(c):
        return (COST_RATIO * c["missed"] + c["false_alarms"]) / c["n"]
    d = [cost(per_station[k][a]) - cost(per_station[k][b]) for k in per_station]
    better = sum(x < 0 for x in d); worse = sum(x > 0 for x in d)
    p = binomtest(better, better + worse, 0.5).pvalue if better + worse else np.nan
    return better, worse, float(p)


# ---- main ----

def run(table, fsets, train_years, test_years, rng, n_boot, delay, kind):
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
            p = make_model(kind).fit(train[cols], train["y"]).predict_proba(test[cols])[:, 1]
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
        better, worse, p = sign_test(per_station, a, b)
        ci.append({"delay_days": delay, "comparison": f"{a} - {b}", "cost_diff": mean,
                   "ci_low": lo, "ci_high": hi, "excludes_zero": bool(lo > 0 or hi < 0),
                   "stations_better": better, "stations_worse": worse,
                   "sign_test_p": p})
    return folds, summary, pd.DataFrame(ci)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-years", default="2023")
    ap.add_argument("--test-years", default="2024")
    ap.add_argument("--delays", nargs="+", type=int, default=[0])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weather", default=None,
                    help="archived weather forecasts (weather_history.parquet); "
                         "restricts every model to days that have one")
    ap.add_argument("--model", choices=["logistic", "logistic_wx", "hgb"], default="logistic",
                    help="logistic on pollution features (default) or the "
                         "gradient-boosting model used for the hourly forecast")
    ap.add_argument("--obs-dir", default=str(OBS_DIR),
                    help="observation store to read (default %(default)s)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output files, e.g. smoke_2019_20")
    args = ap.parse_args()
    train_years, test_years = parse_years(args.train_years), parse_years(args.test_years)
    if set(train_years) & set(test_years):
        ap.error("training and test years must not overlap")
    rng = np.random.default_rng(args.seed)

    print(f"Model: {args.model}\nLoading observations...")
    daily, sites = load_daily(args.obs_dir)
    weather = load_weather(args.weather) if args.weather else None
    if args.model == "logistic_wx" and weather is None:
        ap.error("--model logistic_wx needs --weather")
    print(f"  {len(daily):,} station-days, {daily['site_id'].nunique()} stations, "
          f"{daily['day'].min().date()} to {daily['day'].max().date()}")

    all_folds, all_summ, all_ci = [], [], []
    for delay in args.delays:
        table, fsets = build_table(daily, sites, delay, weather)
        tr = table[table["year"].isin(train_years)]
        te = table[table["year"].isin(test_years)]
        print(f"\n=== delay {delay} day(s): features from day T-{1 + delay} ===")
        print(f"  train {train_years[0]}-{train_years[-1]}: {len(tr):,} station-days, "
              f"{int(tr['y'].sum())} over 25 on {tr.loc[tr['y'] == 1, 'day'].nunique()} dates")
        print(f"  test  {test_years[0]}-{test_years[-1]}: {len(te):,} station-days, "
              f"{int(te['y'].sum())} over 25 on {te.loc[te['y'] == 1, 'day'].nunique()} dates")
        folds, summ, ci = run(table, fsets[args.model], train_years, test_years,
                              rng, args.n_boot, delay, args.model)
        all_folds.append(folds); all_summ.append(summ); all_ci.append(ci)

        show = summ[summ["test_years"] == "all"]
        print("\n  " + show[["method", "events", "caught", "false_alarms", "recall",
                             "precision", "cost"]].round(4).to_string(index=False)
              .replace("\n", "\n  "))
        print("\n  Cost difference, 95% CI over stations (negative = first is better);")
        print("  stations where first is better/worse, sign-test p:")
        for r in ci.itertuples():
            print(f"    {r.comparison:<45} {r.cost_diff:+.4f}  [{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
                  f"{'  *' if r.excludes_zero else '   '}  {r.stations_better:>2}/{r.stations_worse:<2} p={r.sign_test_p:.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = "daily_target" + (f"_{args.tag}" if args.tag else "")
    pd.concat(all_folds).to_csv(RESULTS_DIR / f"{stem}_folds.csv", index=False)
    pd.concat(all_summ).to_csv(RESULTS_DIR / f"{stem}_summary.csv", index=False)
    pd.concat(all_ci).to_csv(RESULTS_DIR / f"{stem}_ci.csv", index=False)
    (RESULTS_DIR / f"{stem}.json").write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "train_years": train_years, "test_years": test_years, "delays": args.delays,
        "threshold": THRESHOLD, "min_valid_hours": MIN_HOURS,
        "cost_ratio": COST_RATIO, "decision_threshold": TAU,
        "validation": "hold out each station AND the test years; train on other "
                      "stations in the training years only",
        "model": args.model,
        "features": fsets[args.model],
        "n_bootstrap": args.n_boot,
    }, indent=2))
    print(f"\nSaved to {RESULTS_DIR}/{stem}_*.csv and {stem}.json")


if __name__ == "__main__":
    main()