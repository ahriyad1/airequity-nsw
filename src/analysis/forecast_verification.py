"""Compare the live daily forecasts with what the stations recorded.
"""
import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.daily_target import MIN_HOURS, THRESHOLD  # noqa: E402

FC = "data/processed/latest_daily_forecast.parquet"
OBS = "data/processed/latest_observations.parquet"
OUT = Path("results/forecast_verification.csv")


def versions(path):
    commits = subprocess.run(["git", "log", "--format=%h", "--", path],
                             capture_output=True, text=True, check=True).stdout.split()
    for c in commits:
        blob = subprocess.run(["git", "show", f"{c}:{path}"], capture_output=True)
        if blob.returncode == 0:
            yield pd.read_parquet(io.BytesIO(blob.stdout))


fc = pd.concat(versions(FC), ignore_index=True)
fc = fc.sort_values("issued_at").drop_duplicates(["site_id", "target_day", "horizon"], keep="last")

obs = pd.concat(versions(OBS), ignore_index=True).drop_duplicates(["site_id", "timestamp"])
obs["PM2.5"] = obs["PM2.5"].clip(lower=0)
obs["day"] = obs["timestamp"].dt.floor("D")
daily = obs.groupby(["site_id", "day"])["PM2.5"].agg(observed="mean", hours="count").reset_index()
daily = daily[daily["hours"] >= MIN_HOURS]

m = fc.merge(daily, left_on=["site_id", "target_day"], right_on=["site_id", "day"], how="inner")
m["bad"] = m["observed"] > THRESHOLD
m.to_csv(OUT, index=False)

print(f"Forecasts checked: {len(m)} (target days {m['target_day'].min().date()} "
      f"to {m['target_day'].max().date()})")
for h, g in m.groupby("horizon"):
    print(f"  {h:8s} n={len(g):3d}  bad days={int(g['bad'].sum())}  "
          f"advisories={int(g['alert_with_sensor'].sum())}  "
          f"caught={int((g['bad'] & (g['alert_with_sensor'] == 1)).sum())}  "
          f"highest chance={g['prob_with_sensor'].max():.1%}  "
          f"highest observed={g['observed'].max():.1f}")
print(f"Saved to {OUT}")