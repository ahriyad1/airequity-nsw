"""Which inputs drive the daily forecast? (A3 feedback: feature importance)
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.daily_target import TAU, build_table, load_daily, make_model  # noqa: E402

HISTORY = Path("data/processed/history/observations")
OUT = Path("results/feature_importance.json")

GROUPS = {
    "Own sensor: evening mean": ["pm_eve"],
    "Own sensor: base day mean, max, min": ["pm_mean", "pm_max", "pm_min"],
    "Own sensor: earlier days": ["pm_mean_lag1", "pm_mean_lag2", "pm_mean_3d"],
    "Neighbours: base day": ["net_mean", "net_max", "net_n_over", "net_idw", "net_nearest"],
    "Neighbours: previous day": ["net_mean_lag1", "net_idw_lag1"],
}

daily, sites = load_daily(HISTORY)
result = {"weights": {}, "permutation_2019_20": {}}

for delay in (0, 1):
    table, fsets = build_table(daily, sites, delay)
    for cond, cols in fsets["logistic"].items():
        model = make_model("logistic").fit(table[cols], table["y"])
        w = dict(zip(cols, np.round(model[-1].coef_[0], 3).tolist()))
        result["weights"][f"delay{delay}_{cond}"] = w
        print(f"weights, readings {delay + 1} day(s) old, {cond}:")
        for c, v in sorted(w.items(), key=lambda kv: -kv[1]):
            print(f"   {c:15s} {v:6.3f}")

table, fsets = build_table(daily, sites, 0)
year = table["day"].dt.year
train = table[year.isin([2016, 2017, 2018, 2021, 2022, 2023])]
test = table[year.isin([2019, 2020])]
rng = np.random.default_rng(42)
for cond, cols in fsets["logistic"].items():
    model = make_model("logistic").fit(train[cols], train["y"])
    X, y = test[cols], test["y"].to_numpy()
    p = model.predict_proba(X)[:, 1]
    res = {"bad_days": int(y.sum()), "caught": int(((p >= TAU) & (y == 1)).sum()),
           "avg_precision": round(float(average_precision_score(y, p)), 3), "shuffled": {}}
    for name, g in GROUPS.items():
        if not set(g) <= set(cols):
            continue
        caught, ap = [], []
        for _ in range(20):
            Xp = X.copy()
            Xp[g] = X[g].to_numpy()[rng.permutation(len(X))]
            pp = model.predict_proba(Xp)[:, 1]
            caught.append(((pp >= TAU) & (y == 1)).sum())
            ap.append(average_precision_score(y, pp))
        res["shuffled"][name] = {"caught": round(float(np.mean(caught)), 1),
                                 "avg_precision": round(float(np.mean(ap)), 3)}
    result["permutation_2019_20"][cond] = res
    print(f"\npermutation, 2019-20, {cond}: caught {res['caught']} of {res['bad_days']}")
    for name, r in res["shuffled"].items():
        print(f"   shuffle {name:38s} caught {r['caught']:6.1f}")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(result, indent=1))
print(f"\nSaved to {OUT}")