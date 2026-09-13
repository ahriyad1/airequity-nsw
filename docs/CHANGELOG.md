# Change Log

Documented changes to AirEquity since Assessment 1. Each entry records what
changed, why, and the evidence supporting the decision.

---

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
