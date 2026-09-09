
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

def feature_sets():

def make_model():

def metrics():

def fit_predict():

def threshold_sweep():

def main():


if __name__ == "__main__":
    main()