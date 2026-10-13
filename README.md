# AirEquity

**Threshold-crossing air quality forecasting for unmonitored communities in Sydney**

PRT661 Data Science Practice · Charles Darwin University · Semester 2, 2026
Theme 2: Predictive Analytics and Forecasting

**Live dashboard:** https://airequity-sydney.streamlit.app/

---

## What this project does

Sydney has 18 operational PM2.5 monitoring stations for about 4.9 million
people. Half of residents live more than 5 km from the nearest one, so during
smoke events or pollution build-ups most people have no local reading to act on.

AirEquity forecasts **the chance that PM2.5 will go over 25 µg/m³ in the next
24 hours** at each station, with and without that station's own sensor. The
gap between the two tells us how much less reliable a forecast is for a place
with no monitor, which is the situation most residents are in.

**Decision supported:** should a health advisory be issued for this area tomorrow?

The system is decision support only. It does not replace official NSW health advice.

---

## Current status (October 2026)

| Component                                          | Status                                    |
| -------------------------------------------------- | ----------------------------------------- |
| Data ingestion (NSW Air Quality API, Open-Meteo)   | Running daily                             |
| Storage (Parquet, DuckDB)                          | Done                                      |
| Temporal and spatial features                      | Done                                      |
| Forecast model and daily batch forecast            | Running daily, under revision (see below) |
| Dashboard                                          | Live at the link above                    |
| Automation (GitHub Actions) and data quality tests | Running daily                             |
| Coverage and equity analysis (ABS SEIFA)           | Done                                      |

**Known limitation.** The A2 validation held out one station at a time but
trained on the other stations over the same days it tested on. A stricter
check that also separates time (`src/analysis/leakage_check.py`) shows the
hourly model has little real skill on unseen days: mean recall drops from
about 13% to 3%, below simple persistence (about 10%). The A2 headline figures
are therefore optimistic. For A4 we are testing a daily-average target and
training on more years (2016–2025) to see whether skill can be recovered.
Details are in `docs/CHANGELOG.md`.

---

## Repository structure

```
.github/workflows/
    pipeline.yml            daily forecast: rebuild, test, fetch, predict, publish
    fetch_history.yml       one-off fetch of historical years into a data release
src/
    ingest/                 NSW Air Quality API and Open-Meteo clients
    storage/                raw JSON -> partitioned Parquet
    features/               temporal features (build_features.py), spatial features (spatial_features.py)
    models/                 baselines, leave-one-station-out validation, bootstrap CIs, daily forecast
    analysis/               leakage check, data delay, coverage and equity, daily-target experiment
    dashboard/              Streamlit app
tests/                      data quality tests (run before every forecast)
results/                    CSV/JSON outputs of the validation and analyses
docs/                       change log and short findings notes
diagrams/                   architecture diagram (.drawio and .png)
data/                       not committed, except the three small forecast files the dashboard reads
```

---

## Data sources

| Source                                                    | Content                                                     | Access        |
| --------------------------------------------------------- | ----------------------------------------------------------- | ------------- |
| [NSW Air Quality API](https://data.airquality.nsw.gov.au) | Hourly PM2.5, temperature, humidity, wind; station metadata | Free, no key  |
| [Open-Meteo](https://open-meteo.com)                      | Recent and forecast weather                                 | Free, no key  |
| ABS SEIFA 2021 (SA2)                                      | Socio-economic disadvantage by area                         | Free download |

No source needs credentials. Never commit keys or tokens to this repository.

Historical observations are stored as GitHub releases rather than in git:
`v0.1-data` (2023–2024, used by the daily forecast) and `v0.2-data`
(2016–2025, built by `fetch_history.yml` for the A4 experiments).

---

## Setup

```bash
git clone https://github.com/ahriyad1/airequity-nsw.git
cd airequity-nsw
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the pipeline

These are the same steps the daily GitHub Actions workflow runs.

```bash
# 1. Historical observations (2023-2024) from the data release
curl -sSL -o obs.zip https://github.com/ahriyad1/airequity-nsw/releases/download/v0.1-data/observations.zip
unzip -q obs.zip -d data/processed && rm obs.zip

# 2. Station list, then features
python src/ingest/fetch_observations.py --list-sites
python src/features/build_features.py
python src/features/spatial_features.py

# 3. Data quality tests
python -m pytest tests/ -v

# 4. Recent data and tomorrow's forecast
python src/ingest/fetch_observations.py --start 2026-10-01 --end 2026-10-12 --parameters PM2.5 TEMP HUMID WSP WDR
python src/storage/write_parquet.py
python src/ingest/fetch_weather_openMeteo.py --past-days 3
python src/models/predict_today.py

# 5. Dashboard
streamlit run src/dashboard/app.py
```

Validation and analyses:

```bash
python src/models/loso.py                 # leave-one-station-out validation
python src/models/bootstrap_ci.py         # confidence intervals for the sensor gap
python src/analysis/leakage_check.py      # same-period vs earlier-period training
python src/analysis/data_delay.py         # effect of late-arriving observations
python src/analysis/coverage_equity.py    # distance to nearest station vs SEIFA
python src/analysis/daily_target.py       # daily-average target, time + station held out
```

---

## Automation

`.github/workflows/pipeline.yml` runs every day at 21:00 UTC (7–8am Sydney) and
whenever code in `src/` or `tests/` changes. It rebuilds the training data,
runs the data quality tests (the forecast stops if any fail), fetches the last
ten days of observations and the weather forecast, produces the next 24 hours of
forecasts, and commits three small files that the live dashboard reads.

---

## Method

1. **Baselines first.** Persistence (tomorrow looks like today) and predicting
   no exceedance. A model that cannot beat them is reported as such.
2. **Threshold target.** The label is PM2.5 above 25 µg/m³, not a concentration.
3. **Spatial features.** Inverse-distance and nearest-neighbour readings,
   upwind stations and network summaries, always excluding the station itself.
   These are what make a forecast possible where there is no sensor.
4. **Cost-weighted decisions.** A missed advisory is treated as 10 times worse
   than a false alarm, and alarm thresholds are set accordingly.
5. **Validation.** Each station is held out in turn and forecast from the
   others. From A4, the test years are also held out from training.
6. **Equity.** Distance to the nearest station is compared with ABS SEIFA
   disadvantage for 331 Sydney SA2 areas.

Metrics: recall, precision, average precision and cost per station-hour or
station-day. Plain accuracy is not used: bad-air periods are about 1% of the
data, so "always fine" would score 99% while warning nobody.

---

## Coverage and equity findings

- 50% of residents live more than 5 km from a PM2.5 station, 13% more than
  10 km and 2% more than 20 km.
- More disadvantaged areas are, on average, slightly _closer_ to a station
  (Spearman ρ = 0.23 between SEIFA score and distance, n = 331 SA2s).
- Residents per station differ about 2.4 times between regions: about 391,000
  per station in Sydney East against 166,000 in Sydney South-west.

Outputs are in `results/coverage_*.csv` and `results/coverage_equity_summary.json`.

---

## Ethics and governance

- **Environmental justice.** If forecasts are weaker where monitoring is
  sparse, the people with least information get the least reliable warnings.
  This is tested directly with the with/without-sensor comparison and SEIFA.
- **Asymmetric errors.** A missed advisory can cause respiratory harm; a false
  alarm causes inconvenience. The 10:1 cost ratio reflects this and is stated
  openly.
- **Honest reporting.** When a stricter test showed earlier results were
  optimistic, we reported it and changed the validation, rather than keeping
  the better-looking numbers.
- **Stale data.** The dashboard shows when its data was last updated.
- **Scope.** Decision support only; official NSW advisories take precedence.

---

## Team

| Member   | Component ownership                                          |
| -------- | ------------------------------------------------------------ |
| _[Name]_ | Acquisition: API clients, resilience, schema validation      |
| _[Name]_ | Storage and processing: Parquet, DuckDB, feature engineering |
| _[Name]_ | Analytics/ML: baselines, classifier, validation              |
| _[Name]_ | Visualisation: Streamlit dashboard                           |
| _[Name]_ | Automation and governance: GitHub Actions, tests, ethics     |

---

## Assessments

|     | Focus                                    | Status      |
| --- | ---------------------------------------- | ----------- |
| A1  | Project proposal and design              | Submitted   |
| A2  | Progress report and development          | Submitted   |
| A3  | Technical demonstration and presentation | Submitted   |
| A4  | Final professional report                | In progress |

Changes since A1 are recorded in `docs/CHANGELOG.md`.
