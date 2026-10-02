# Change Log

Documented changes to AirEquity since Assessment 1, newest first. Each entry records what
changed, why, and the evidence supporting the decision.

---

# Changes since Assessment 2

Newest section first. Each entry records what changed, why, and where the
evidence is. Commit references are short hashes in the project repository.

---

## Forecasting

### Batch forecasting added

**Previous state.** The model had been validated but produced no forecasts.
Nothing in the repository could answer "what is the chance of exceeding
25 µg/m³ at this station tomorrow?"

**Current state.** `src/models/predict_today.py` produces a 24-hour forecast
for all 18 operational stations and writes it to a small file the dashboard
reads. It fits two models, one with each station's own sensor history and one
without, and applies the cost-optimal advisory thresholds from the A2
threshold sweep: 6.5% with a sensor and 9% without.

**Why.** Two forecasts per station show the project's central finding in the
running system, not only in the report. Forecasts are computed in a batch
rather than inside the dashboard because fitting on 277,149 rows would exceed
the memory of a Streamlit Community Cloud container.

**Evidence.** `6f2231a`

---

### Forecast inputs built from recent observations

**Previous state.** The first version of the forecast took its neighbour
readings from the last row of the training table, which ends in December 2024. Forecasts for September 2026 were combining real weather with
neighbour readings twenty months old.

**Current state.** Neighbour features and each station's own history are
rebuilt from the most recent observations, using the same functions as the
training pipeline. The output records how old those observations are.

**Why.** The stale readings were producing false alarms. With them, Richmond
showed a 25.1% chance and four hours on advisory; with current readings no
station passed 5.2%, consistent with network readings of 3 to 5 µg/m³ that week.

**Evidence.** `6f2231a`

---

### Forecast hours aligned with the training horizon (correction)

**Previous state.** The model is trained on features at time _t_ predicting
PM2.5 at _t_ + 24 hours. The first forecast script supplied each future hour's
weather and labelled the result as the forecast for that same hour, so every
forecast was shown a day early.

**Current state.** To forecast target hour _T_, features are built for
_T_ − 24 hours, which falls within the past day, exactly as in training. The
output stores both the feature hour and the target hour so the alignment can
be checked. The weather fetch now requests three days of history so the
24-hour weather lag is always covered.

**Why.** Found while planning a follow-up analysis. Without the fix, the
dashboard would have shown correct probabilities against the wrong hours.

**Evidence.** `ab23d6f`

---

## Data sources

### Open-Meteo weather ingestion added

**Previous state.** Open-Meteo was listed in A1 and A2 as a planned source but
no client existed.

**Current state.** `src/ingest/fetch_weather_openMeteo.py` fetches hourly
weather for every station with coordinate validation, retries, and wind speed
requested in m/s to match the NSW observations. It writes a raw JSON archive,
a dated Parquet snapshot and a latest-copy file.

**Why.** NSW station readings reach the API more than a day after they are
taken, so the forecast needs another source for recent weather.

**Evidence.** `src/ingest/fetch_weather_openMeteo.py`

---

## Automation and testing

### Working daily pipeline replaces the placeholder workflow

**Previous state.** `.github/workflows/pipeline.yml` referenced scripts that
did not exist and failed on every scheduled run.

**Current state.** The workflow runs daily and on demand. It rebuilds the
2023–2024 training data from the project's data release, runs the data
quality tests, fetches recent observations and weather, produces the forecast
and commits the forecast files so the live dashboard updates.

**Why.** Completes the workflow automation component, the only one of the
six that was not operational at A2.

**Evidence.** `c3048e4`, `7a4c0fc`, `092756e`

---

### Data quality tests run as a gate

**Previous state.** Tests existed but were run by hand.

**Current state.** The pipeline stops before producing a forecast if any test
fails, so bad data cannot reach the dashboard. The suite has two layers:
observations may contain small negative values from instrument noise, while
the feature table must not.

**Why.** On the first automated runs the gate did exactly this, stopping the
pipeline when unclipped negative values reached the features (see regression
below).

**Evidence.** `tests/test_data_quality.py`; commit "Restore two-layer data
quality tests; update action versions"

---

### Model parity on the pipeline runner

**Previous state.** Scripts prefer LightGBM when it is installed and fall
back to scikit-learn otherwise.

**Current state.** The pipeline uninstalls LightGBM after installing
dependencies.

**Why.** LightGBM installs cleanly on GitHub's Linux runners. Without this
step the published forecasts would come from a different model than the one
validated and reported, which used scikit-learn's HistGradientBoosting.

**Evidence.** `c3048e4`

---

### Regression found and repaired

**Previous state.** Three fixes made before A2 were in place: negative PM2.5
clipped to zero in the feature table, and both feature scripts clearing their
output folders before writing.

**What happened.** Older versions of `build_features.py` and
`spatial_features.py` were committed over the fixed ones, undoing all three.
The first full pipeline run produced 20,322 negative values in the features,
and a local rebuild produced 556,769 rows where 277,155 were expected.

**Current state.** All three fixes restored. The tests that caught the
problem now run on every pipeline run.

**Why it matters.** The fixes were silently lost and only found because the
tests ran automatically. Merges into `main` should be checked with
`git diff main <branch>` before they are made.

**Evidence.** `092756e`

---

### Training restricted to 2023–2024

**Previous state.** The feature script trained on every observation in the
store.

**Current state.** Training is limited to 1 January 2023 to 31 December 2024.

**Why.** The observations store now also holds recent weeks fetched for
forecasting. Without the limit, those weeks would enter the training data and
the running model would no longer match the one reported. The rebuilt table
has 277,149 rows against 277,155 at A2; the six rows lost fall on the final
day of December, and the 3,166 exceedances are unchanged.

**Evidence.** `092756e`

---

## Interface

### Dashboard redesigned around the forecast

**Previous state.** The dashboard showed historical observations, crossing
rates and the September 2023 smoke event. It did not show forecasts.

**Current state.** Four tabs: Forecast (headline advisory status, a 24-hour
station-by-hour grid, a toggle between forecasts with and without each
station's own sensor, and a station ranking); Station (recent readings and
both forecasts); Network history (the original views); and Method (how the
forecast is made and its limitations).

**Why.** The primary user from A1, a council environmental health officer,
needs the advisory decision first and the history second.

**Evidence.** `574eed7`

---

### Forecast files published for the live site

**Previous state.** Everything under `data/` was excluded from Git, so the
deployed dashboard could not see any forecast.

**Current state.** Three small files are tracked as exceptions:
`latest_forecast.parquet`, `latest_forecast_meta.json` and
`latest_observations.parquet`. The pipeline commits new versions each run.

**Evidence.** `4e69f8b`

---

## Analysis

### Per-capita coverage and SEIFA analysis

**Previous state.** A1 promised to test whether coverage was uneven relative
to population and whether forecast quality tracked disadvantage. Neither had
been done at A2.

**Current state.** `src/analysis/coverage_equity.py` uses ABS SEIFA 2021 by
SA2 for 343 areas in the Sydney basin (4.88 million residents, 2021 Census).
Each area is assigned to its nearest operational station.

**Findings.**

- Half of residents live within 5 km of a monitor; 12.6% live more than
  10 km away and 2.1% more than 20 km.
- Stations per 100,000 residents: Sydney East 0.26, North-west 0.46,
  South-west 0.60.
- More disadvantaged areas are closer to monitors, not further
  (Spearman ρ = +0.229, p < 0.0001, 331 areas).
- More disadvantaged station catchments have higher crossing rates
  (ρ = −0.395, p = 0.104, 18 stations; indicative only).

**Evidence.** `c851440`; `results/coverage_*.csv`, `results/station_catchments.csv`

---

## Corrections to Assessment 2

### Coverage premise not supported per capita

**A2 stated.** Populations facing the greatest exposure have the least direct
measurement (§19.1, §24).

**Corrected.** Per resident, western Sydney is better covered than the east,
and disadvantaged areas sit closer to monitors. The equity problem is
exposure, not access to monitoring: the more disadvantaged west has more
threshold crossings. The regions in this analysis are station catchments,
which places the Northern Beaches, North Shore and Sutherland in Sydney East
because their nearest station is there.

---

### Operational limitation restated

**A2 stated.** Models were evaluated on recorded weather, and operational
forecasting would use forecast weather, lowering accuracy (§25.2).

**Corrected.** The model uses weather at the time the forecast is made, not
future weather, so forecast weather is not the gap. The gap is reporting
delay: station readings arrive about 40 hours late, so in practice the
neighbour and station inputs are older than they were in testing.

---

## In progress

### Cost of reporting delay

`src/analysis/data_delay.py` is written and tested on synthetic data. It
trains on 2023, tests on 2024, and replaces the station-derived inputs with
their values from 12 to 72 hours earlier to measure the accuracy lost. To be
run and reported before Assessment 4.

---

# Assessment 1 to Assessment 2

## Data characterisation

### PM2.5 frequency variants — correction to A1

**Previous state.** A1 §4 stated that the NSW API returns PM2.5 only as a
24-hour rolling average, and that the project would therefore model a
smoothed series.

**Current state.** Both an hourly average and a 24-hour rolling average are
available. The hourly average is used as the modelling target; the rolling
variant is retained as a candidate feature.

**Why it changed.** The A1 characterisation was based on a single Swagger
test response, which happened to return the rolling variant. Querying the
frequency field across all parameters showed that PM2.5, PM10 and ozone each
return multiple variants:

```
PM2.5    {'Hourly average', '24h rolling average derived from 1h average'}
OZONE    {'Hourly average', '4h rolling', '8h rolling'}
TEMP     {'Hourly average'}
```

**Why it matters.** Lag features at t−1, t−24 and t−168 are only meaningful
against raw hourly observations. On a rolling average, a one-hour lag
describes a smoothed window rather than a distinct measurement.

**Evidence.** `src/ingest/fetch_observations.py`; frequency audit in commit
history.

---

### Operational network size — 137 registry entries, 18 reporting stations

**Previous state.** A1 §4 referred to 137 site records returned by the API,
noting that this included aggregates and test sites.

**Current state.** The operational Sydney network for PM2.5 during 2023–2024
is **18 stations**.

| Filter                                              | Count |
| --------------------------------------------------- | ----- |
| All API entries                                     | 137   |
| Excluding region aggregates, test sites, non-Sydney | 24    |
| Actually reporting PM2.5 in 2023–2024               | 18    |

**Why it changed.** The `get_SiteDetails` endpoint is a registry of
locations, not an inventory of operational monitors. It includes region-level
aggregates with IDs above 1,000,000, a literal "Test Site", and incident
monitoring pods. Five further Sydney stations — Lindfield, Chullora, Bargo,
Vineyard and Macarthur — appear in the registry but returned no PM2.5 data
in any year tested (2018, 2020, 2022, 2024, 2025, 2026). Ultimo-UTS returns
rows with null values throughout.

**Why it matters.** Leave-one-station-out validation runs across 18 folds,
not 24 or 137. Any figure quoted for network size must be the operational
count.

**Evidence.** `docs/station_coverage_finding.md`;
`filter_real_stations()` in `src/ingest/fetch_observations.py`.

---

### Record start dates — ragged across the network

**Previous state.** A1 stated that per-parameter start dates were being
confirmed in Week 1.

**Current state.** Confirmed. PM2.5 availability ranges from 1998 at
Liverpool to 2016 at Bringelly. The common window across all stations begins
in 2016; the modelling period is 2023–2024.

**Why it matters.** Lag features require continuous history. Stations with
different start dates cannot be pooled naively, so the hourly index is
regularised per station before features are computed.

**Evidence.** `docs/data_availability.md`;
`src/ingest/verify_availability.py`.

---

## Scope

### Modelling scope narrowed

**Previous state.** A1 described forecasting across NSW.

**Current state.** The initial implementation is limited to **PM2.5 in the
Sydney basin**, 18 stations, a 24-hour horizon, and a 25 µg/m³ threshold.

**Why it changed.** Responding to A1 feedback recommending a tighter initial
scope. Regional stations were also found to have inconsistent PM2.5
availability — Wagga Wagga returned no PM2.5 records in any year tested —
making a statewide first version unreliable.

---

### Threshold selection

**Previous state.** A1 referred generally to health-category exceedance.

**Current state.** Primary threshold **25 µg/m³** hourly, with 62.1 µg/m³
retained as a secondary severity label.

**Why.** 25 µg/m³ is the national PM2.5 daily standard and the level at which
enHealth guidance rates air as poor or worse. Applied hourly it yields 3,166
positive cases (1.14%) across the study period — rare but trainable. NSW's
interim one-hour threshold of 62.1 µg/m³ yields only 336 cases (0.12%),
insufficient for reliable training across 18 stations.

**Evidence.** `results/baseline_config.json`.

---

## Technology

### LightGBM replaced with scikit-learn HistGradientBoosting

**Previous state.** A1 §7 listed LightGBM as the planned classifier.

**Current state.** `HistGradientBoostingClassifier` from scikit-learn, with
LightGBM retained as an automatic preference where available.

**Why it changed.** LightGBM requires the `libomp` OpenMP runtime, which is
not present on the development machine and cannot be installed without
Homebrew. Both are gradient-boosted tree implementations with equivalent
capability for this task.

**Evidence.** `make_model()` in `src/models/loso_validation.py`; recorded in
`results/loso_config.json`.

---

## Method corrections

### Duplicate partitions on repeated runs

**Problem.** `to_parquet(partition_cols=...)` appends rather than overwrites.
Re-running the feature scripts after an edit produced 554,310 rows — exactly
double the true 277,155 — with each row present twice. This inflated event
counts and placed duplicates in both training and test folds.

**Fix.** Both feature scripts now remove the output directory before writing.

**Evidence.** `src/features/build_features.py`,
`src/features/spatial_features.py`.

---

### Double correction for class imbalance

**Problem.** Class imbalance was being corrected twice — once through
`class_weight="balanced"` during training and again through a Bayes-optimal
decision threshold of 0.091. The combined effect was severe overcorrection:
the model flagged roughly 40% of all hours, achieving recall of 0.93 at
precision of 0.019 and a cost-weighted loss of 0.54, against 0.114 for
predicting nothing.

**Fix.** Class weighting removed. Imbalance is now handled solely at the
decision threshold, keeping the cost logic explicit and in one place.

**Result after correction.** Recall 0.37 at precision 0.35, cost-weighted
loss 0.083 — a 27% improvement over the always-negative baseline.

**Evidence.** `make_model()` in `src/models/loso_validation.py`;
`results/threshold_sweep_*.csv`.

---

### Decision threshold determined empirically

**Previous state.** The Bayes-optimal threshold τ = 1/(1+cost_ratio) = 0.091
was adopted on theoretical grounds.

**Current state.** A threshold sweep across 0.005–0.50 identifies the
cost-minimising operating point: **0.07 monitored, 0.09 unmonitored**.

**Why it changed.** The theoretical value assumes calibrated probabilities.
Sweeping empirically confirms where the model actually minimises cost, and
produces a trade-off curve rather than a single point.

**Note.** The sweep is fitted on the same folds it is evaluated on, so the
selected threshold is mildly optimistic. This is stated rather than adjusted
for.

---

## Findings that revised the problem framing

### Equity claim now supported by exposure data

**Previous state.** A1 §1 noted station counts by region and stated that
coverage relative to population was being assessed.

**Current state.** Threshold crossing rates differ materially by region:

| Region            | Stations | Crossing rate |
| ----------------- | -------- | ------------- |
| Sydney North-west | 6        | 1.60%         |
| Sydney South-west | 5        | 1.06%         |
| Sydney East       | 7        | 0.81%         |

Prospect records 5.3× the crossing rate of Randwick. The five highest-rate
stations are all in western Sydney.

**Why it matters.** The information gap is largest where exposure is
highest — a stronger claim than station counts alone, and measured rather
than assumed.

**Evidence.** `docs/exposure_by_region.md`.

---

### Reliability gap smaller than hypothesised

**Previous state.** A1 hypothesised that forecast reliability would degrade
at unmonitored locations, and that the degradation would be measurable.

**Current state.** Across 18 LOSO folds, mean recall is 0.370 with the
station's own sensor history and 0.376 without — a difference of −1.7%, well
within fold-to-fold variation. Per-station gaps scatter symmetrically about
zero and show no correlation with distance to the nearest station
(r = 0.030).

**Interpretation.** For 24-hour-ahead threshold forecasting in the Sydney
basin, a station's own recent history adds little beyond what surrounding
stations and meteorology already provide. Threshold crossings appear to be
driven by regional forcing rather than local persistence — consistent with
the persistence baseline achieving only 0.10 recall.

**Implication.** The absence of a reliability penalty strengthens rather
than weakens the project's motivation: forecasts at unmonitored locations
are approximately as reliable as at monitored ones, so the information gap
reflects service provision rather than a modelling limitation.

**Caveat.** With 78–406 events per fold, individual station gaps are noisy;
only the pooled mean is reliable.

**Evidence.** `results/loso_results.csv`,
`results/loso_reliability_gap.csv`.

---

## Team

### Group size increased from four to five

**Previous state.** A1 documented a four-member team.

**Current state.** Five members. Task allocation redistributed accordingly,
including in response to A1 feedback that technical responsibilities were
concentrated on one member.

**Evidence.** Jira board; A2 §9 task allocation table.
