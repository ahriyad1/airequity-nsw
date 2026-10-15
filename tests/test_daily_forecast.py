"""
Checks for the daily forecast (src/analysis/daily_target.py and
src/models/predict_daily.py)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis.daily_target import (
    NonNegLogistic, TAU, build_table, daily_features, load_daily, make_model)

OBSERVATIONS = Path("data/processed/observations")
FORECAST = Path("data/processed/latest_daily_forecast.parquet")
FORECAST_META = Path("data/processed/latest_daily_forecast_meta.json")


@pytest.fixture(scope="module")
def daily():
    if not OBSERVATIONS.exists():
        pytest.skip("observations not built")
    return load_daily(OBSERVATIONS)


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_features_come_from_earlier_days(daily, delay):
    table, _ = build_table(*daily, delay)
    gap = (table["day"] - table["base_day"]).dt.days
    assert (gap == 1 + delay).all()


def test_network_features_exclude_the_station(daily):
    d, sites = daily
    feat, _, _ = daily_features(d, sites)
    rows = feat.dropna(subset=["net_mean"]).sample(25, random_state=0)
    for r in rows.itertuples():
        others = d[(d["day"] == r.base_day) & (d["site_id"] != r.site_id)]["pm_mean"]
        assert np.isclose(others.mean(), r.net_mean)


def test_target_is_the_stations_daily_mean(daily):
    d, sites = daily
    table, _ = build_table(d, sites, 0)
    means = d.set_index(["site_id", "day"])["pm_mean"]
    rows = table.sample(25, random_state=0)
    for r in rows.itertuples():
        assert np.isclose(means.loc[(r.site_id, r.day)], r.target)
        assert r.y == int(r.target > 25)


def test_gaps_filled_from_available_days(daily):
    d, sites = daily
    feat, _, _ = daily_features(d, sites)
    present = feat["pm_mean"].notna()
    assert feat.loc[present, ["pm_mean_lag1", "pm_mean_lag2", "pm_mean_3d"]].notna().all().all()


def test_weights_never_negative():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 4))
    # one feature truly lowers the chance; the constrained model must not use it negatively
    y = (X @ [1.0, 0.5, -1.0, 0.0] + rng.logistic(size=3000) > 1).astype(int)
    m = NonNegLogistic().fit(X, y)
    assert (m.coef_ >= 0).all()
    p = m.predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_more_pollution_never_lowers_the_chance(daily):
    table, fsets = build_table(*daily, 1)
    cols = fsets["logistic"]["with_sensor"]
    model = make_model("logistic").fit(table[cols], table["y"])
    X = table[cols].dropna().sample(200, random_state=0)
    worse = X * 3 + 20       # every pollution feature higher
    worse["net_n_over"] = X["net_n_over"]  # a count, kept as is
    assert (model.predict_proba(worse)[:, 1] >= model.predict_proba(X)[:, 1] - 1e-9).all()


def test_published_forecast_is_well_formed():
    if not (FORECAST.exists() and FORECAST_META.exists()):
        pytest.skip("no daily forecast published yet")
    fc = pd.read_parquet(FORECAST)
    meta = json.loads(FORECAST_META.read_text())
    for col in ("prob_with_sensor", "prob_without_sensor"):
        p = fc[col].dropna()
        assert ((p >= 0) & (p <= 1)).all()
    thr = meta["decision_threshold"]
    assert np.isclose(thr, TAU, atol=1e-4)
    ok = fc["prob_without_sensor"].notna()
    assert (fc.loc[ok, "alert_without_sensor"] == (fc.loc[ok, "prob_without_sensor"] >= TAU)).all()
    assert set(fc["horizon"]) <= {"today", "tomorrow"}
    assert (fc["target_day"] > fc["base_day"]).all()