# AirEquity

**Threshold-Crossing Air Quality Forecasting for Unmonitored Communities in NSW**

PRT661 Data Science Practice · Charles Darwin University · Semester 2, 2026
Theme 2 — Predictive Analytics and Forecasting

---

## What this project does

NSW operates 21 air quality monitoring stations across Greater Sydney and over 90 statewide. Most residents do not live near one, so during bushfire smoke episodes or pollution build-ups they have no locally relevant information on which to act.

AirEquity forecasts the **probability that air quality will cross a health threshold in the next 24 hours**, including for locations with no monitoring station, and measures how much less reliable those unmonitored predictions are.

**Decision supported:** should a health advisory be issued for a given location in the next 24 hours?

**Core method:** leave-one-station-out (LOSO) validation — hide a real station, predict it blind from neighbouring stations and weather, then compare against its actual recorded readings. The accuracy gap is the measured reliability loss.

---

## Repository structure

```
airequity/
├── README.md
├── requirements.txt
├── .gitignore
├── .github/workflows/         # GitHub Actions — scheduled pipeline
├── data/
│   ├── raw/                   # archived API responses (not committed)
│   └── processed/             # Parquet outputs (not committed)
├── src/
│   ├── ingest/                # API clients, schema validation
│   ├── storage/               # Parquet writers, DuckDB helpers
│   ├── features/              # cleaning, temporal + spatial features
│   ├── models/                # baselines, classifier, LOSO validation
│   └── dashboard/             # Streamlit app
├── notebooks/                 # exploration and analysis
├── tests/
├── diagrams/                  # .drawio source + exported PNG
├── reports/                   # A1–A4 submissions
└── docs/                      # data dictionary, decisions log, ethics register
```

---

## Data sources

| Source | Content | Access | Notes |
|---|---|---|---|
| NSW Air Quality API | Hourly pollutants + meteorology, station metadata, parameter definitions | Free, **no API key** | Pollutants and weather from the same network |
| Open-Meteo | Forecast weather covariates | Free, **no API key** | Provides tomorrow's expected conditions |
| ABS SEIFA | Socioeconomic index by area | Free download | Used only for the equity analysis |

**No credentials are required by any source.** No API keys, tokens, or secrets are to be committed to this repository under any circumstances.

---

## Setup

```bash
git clone <repo-url>
cd airequity

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Running the pipeline

```bash
python -m src.ingest.fetch_observations     # pull latest air quality data
python -m src.ingest.fetch_weather          # pull weather forecasts
python -m src.features.build                # clean and engineer features
python -m src.models.train                  # baselines + classifier
python -m src.models.loso                   # leave-one-station-out validation

streamlit run src/dashboard/app.py          # launch dashboard
```

Automated execution runs on schedule via GitHub Actions. See `.github/workflows/`.

---

## Method

1. **Baselines first** — persistence and seasonal-naive. All models are measured against these. If a model cannot beat them, that is reported as a result.
2. **Threshold reframing** — the target is the probability of exceeding a health category boundary, not a point concentration.
3. **Spatial features** — inter-station distances, upwind station readings, wind-vector alignment. These are what make prediction possible at locations without a sensor.
4. **LOSO validation** — each station held out in turn and predicted blind, giving a per-location reliability measurement.
5. **Equity analysis** — per-station error correlated against monitoring density and ABS SEIFA scores.

**Evaluation metrics:** recall, precision, Brier score, and cost-weighted misclassification loss. Plain accuracy is not used — bad-air days are rare, so a model that always predicts "fine" would score highly while warning nobody.

---

## Team

| Member | Component ownership |
|---|---|
| *[Name]* | Acquisition — API clients, resilience, schema validation |
| *[Name]* | Storage and processing — Parquet, DuckDB, feature engineering |
| *[Name]* | Analytics/ML — baselines, classifier, LOSO validation |
| *[Name]* | Visualisation — Streamlit dashboard, decision interface |
| *[Name]* | Automation and governance — GitHub Actions, tests, monitoring, ethics register |

Task board: *[link]*

---

## Assessments

| | Focus | Due | Status |
|---|---|---|---|
| A1 | Project proposal and design | Week 3 | |
| A2 | Progress report and development | Week 6 | |
| A3 | Technical demonstration and presentation | Week 9 | |
| A4 | Final professional report | Week 12 | |

---

## Ethics and governance

Full register in `docs/ethics_register.md`. Summary:

- **Environmental justice** — if accuracy degrades where monitoring is sparse, communities with greatest exposure receive the least reliable warnings. Tested by correlating error against SEIFA.
- **Asymmetric misclassification** — a missed advisory risks respiratory harm; a false alarm causes inconvenience. Thresholds are cost-weighted accordingly and the choice is documented.
- **Uncertainty disclosure** — predictions for unmonitored locations display a reliability estimate. Confident numbers are never shown for locations that cannot be verified.
- **Fail-safe behaviour** — if an API is unavailable, cached data is served with an explicit staleness warning rather than silently presented as current.
- **Scope** — this system is decision support. It does not replace official NSW health advisories.

---

## Change log

Significant design changes are recorded in `docs/decisions.md` with date and rationale.
## Streamlit Dashboard

AirEquity includes an interactive Streamlit dashboard for exploring PM2.5 exposure across 18 operational monitoring stations in Greater Sydney.

### Dashboard features

- Station selector
- PM2.5 hourly trend chart
- 25 µg/m³ health-threshold reference line
- Interactive Sydney station map
- Station threshold-crossing rates
- Regional exposure comparison
- Smoke-event explorer for 10–14 September 2023

### Regional exposure finding

| Region | Crossing rate |
|---|---:|
| Sydney North-west | 1.60% |
| Sydney South-west | 1.06% |
| Sydney East | 0.81% |

Sydney North-west records approximately twice the threshold-crossing rate observed in Sydney East.

### Run the dashboard

Install dependencies:

```bash
pip install streamlit plotly pandas pyarrow
