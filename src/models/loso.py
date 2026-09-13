
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

def metrics():

def fit_predict():

def threshold_sweep():

def main():


if __name__ == "__main__":
    main()