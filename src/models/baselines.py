"""
Baseline benchmark evaluation for 24-hour PM2.5 threshold-crossing forecasts.

Baseline Models & Benchmark Evaluation

"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_DIR = Path("data/processed/features")
RESULTS_DIR = Path("results")
# Missed event (FN) cost is 10x false alert (FP) cost
DEFAULT_COST_RATIO = 10.0  


def confusion_matrix_counts(y_true, y_pred):
    """Compute true/false positive and negative counts."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def evaluate_model(name, y_true, y_pred, y_prob=None, cost_ratio=DEFAULT_COST_RATIO):
    """Compute operational performance metrics and cost-weighted loss."""
    tp, fp, fn, tn = confusion_matrix_counts(y_true, y_pred)
    n = len(y_true)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    accuracy = (tp + tn) / n
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Alarm Rate

    # Brier score evaluates probability calibration (lower is better)
    prob = y_prob if y_prob is not None else y_pred.astype(float)
    brier = float(np.mean((prob - y_true) ** 2))

    # Asymmetric cost loss normalized per observation
    cost_loss = (cost_ratio * fn + fp) / n

    return {
        "model": name,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "false_alarm_rate": round(far, 4),
        "brier": round(brier, 5),
        "cost_weighted_loss": round(cost_loss, 5),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_climatology_rates(train_df):
    """Compute historical crossing probabilities per station and hour from training data."""
    return (train_df.groupby(["site_id", "hour"])["label"]
                    .mean()
                    .rename("clim_rate")
                    .reset_index())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=25.0,
                        help="PM2.5 threshold in ug/m3 (default: 25.0)")
    parser.add_argument("--cost-ratio", type=float, default=DEFAULT_COST_RATIO,
                        help="Cost of False Negative relative to False Positive (default: 10.0)")
    parser.add_argument("--train-year", type=int, default=2023,
                        help="Training dataset year (default: 2023)")
    parser.add_argument("--test-year", type=int, default=2024,
                        help="Out-of-sample test dataset year (default: 2024)")
    args = parser.parse_args()

    print("Loading engineered features...")
    df = pd.read_parquet(FEATURES_DIR)
    df = df.dropna(subset=["label", "PM2.5", "pm25_lag_24h"])
    df["label"] = df["label"].astype(int)
    print(f"  Loaded {len(df):,} records across {df['site_id'].nunique()} stations.")

    train = df[df["timestamp"].dt.year == args.train_year]
    test = df[df["timestamp"].dt.year == args.test_year]

    print("\nExecuting temporal split (train 2023 -> test 2024):")
    print(f"  Train ({args.train_year}): {len(train):,} rows | {int(train['label'].sum()):,} positive events ({train['label'].mean():.2%})")
    print(f"  Test  ({args.test_year}): {len(test):,} rows | {int(test['label'].sum()):,} positive events ({test['label'].mean():.2%})")

    if train.empty or test.empty:
        raise ValueError("Training or testing partition is empty. Verify dataset year coverage.")

    y_test = test["label"].values
    results = []

    # 1. Zero-Skill Baseline: Always predict 0 (Majority Class)
    results.append(evaluate_model(
        name="always_negative",
        y_true=y_test,
        y_pred=np.zeros(len(y_test), dtype=int),
        y_prob=np.zeros(len(y_test), dtype=float),
        cost_ratio=args.cost_ratio
    ))

    # 2. 24h Persistence: Current reading at forecast issue time t > threshold
    y_pred_persist = (test["PM2.5"].values > args.threshold).astype(int)
    results.append(evaluate_model(
        name="persistence_t0",
        y_true=y_test,
        y_pred=y_pred_persist,
        y_prob=y_pred_persist.astype(float),
        cost_ratio=args.cost_ratio
    ))

    # 3. 48h Lag Persistence: Reading at t-24h > threshold
    y_pred_lag48 = (test["pm25_lag_24h"].values > args.threshold).astype(int)
    results.append(evaluate_model(
        name="lag_48h_prior",
        y_true=y_test,
        y_pred=y_pred_lag48,
        y_prob=y_pred_lag48.astype(float),
        cost_ratio=args.cost_ratio
    ))

    # 4. Stratified Historical Climatology
    bayes_threshold = 1.0 / (1.0 + args.cost_ratio)
    rates = compute_climatology_rates(train)
    merged = test.merge(rates, on=["site_id", "hour"], how="left")
    prob_clim = merged["clim_rate"].fillna(train["label"].mean()).values
    y_pred_clim = (prob_clim >= bayes_threshold).astype(int)

    results.append(evaluate_model(
        name=f"climatology (tau={bayes_threshold:.3f})",
        y_true=y_test,
        y_pred=y_pred_clim,
        y_prob=prob_clim,
        cost_ratio=args.cost_ratio
    ))

    # Summary table output
    res_df = pd.DataFrame(results)
    display_cols = ["model", "recall", "precision", "f1", "false_alarm_rate",
                    "brier", "cost_weighted_loss", "accuracy"]

    print("\n" + "=" * 90)
    print(f"BASELINE BENCHMARK RESULTS — Test Year: {args.test_year} | Threshold: {args.threshold} ug/m3 | Horizon: 24h")
    print("=" * 90)
    print(res_df[display_cols].to_string(index=False))

    print("\nConfusion Matrix Breakdown:")
    print(res_df[["model", "tp", "fp", "fn", "tn"]].to_string(index=False))

    best_model = res_df.loc[res_df["cost_weighted_loss"].idxmin()]
    print(f"\nLowest Cost-Weighted Loss: {best_model['model']} ({best_model['cost_weighted_loss']})")

    # Save artifacts for pipeline lineage
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)

    config_payload = {
        "threshold_ug_m3": args.threshold,
        "horizon_hours": 24,
        "cost_ratio_miss_to_false_alarm": args.cost_ratio,
        "bayes_optimal_threshold": bayes_threshold,
        "train_year": args.train_year,
        "test_year": args.test_year,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_positive_rate": float(test["label"].mean()),
    }
    with open(RESULTS_DIR / "baseline_config.json", "w") as f:
        json.dump(config_payload, f, indent=2)

    print(f"\nResults and configuration archived to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()