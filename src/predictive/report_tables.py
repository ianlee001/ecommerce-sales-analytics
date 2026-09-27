"""
report_tables.py - recompute the Chapter 5 accuracy tables from the saved CV predictions.

The offline pipeline writes per-point predictions to results/:
    lstm_cv_predictions.csv      (LSTM:      series_id, cutoff, ds, h, y_true, y_pred, ...)
    baseline_cv_predictions.csv  (baselines: unique_id, cutoff, ds, h, y, AutoARIMA, ...)

This module reads BOTH, aligns them on the IDENTICAL rolling-origin folds, and reproduces
the thesis Chapter 5.1 accuracy numbers from a committed source file (rather than an
ad-hoc scratchpad script), so every reported figure is traceable and reproducible:

    Table 5.1  pooled (volume-weighted) accuracy of all five models (RMSE / MAE / WAPE / MAPE)
    Table 5.2  per-forecast-step accuracy (LSTM vs AutoARIMA)
    Obs. 3     per-series RMSE + the win counts (LSTM < ARIMA on N/20; LSTM < per-series mean
               on N/20) - the evidence that the pooled RMSE lead is volume-weighted, not a
               majority-of-series win.

Metrics come from src/predictive/metrics.py (the SAME functions the pipeline uses), so
these tables cannot drift from the numbers computed during evaluation.

Run from the repo root:  python -m src.predictive.report_tables
"""
import numpy as np
import pandas as pd

from ..config import RESULTS
from . import metrics as M

BASELINES = ["AutoARIMA", "HistoricAverage", "Naive", "SeasonalNaive"]


def load_merged() -> pd.DataFrame:
    """LSTM + baseline per-point predictions aligned on identical (series, origin, step).

    The two CV files are produced on the same rolling origins (src/config.py), so an inner
    join on (series_id, cutoff, ds, h) recovers exactly the paired points; we assert the
    shared actuals agree so a fold misalignment can never pass silently.
    """
    lstm = pd.read_csv(RESULTS / "lstm_cv_predictions.csv", parse_dates=["cutoff", "ds"])
    base = (pd.read_csv(RESULTS / "baseline_cv_predictions.csv", parse_dates=["cutoff", "ds"])
              .rename(columns={"unique_id": "series_id"}))
    df = lstm.merge(base, on=["series_id", "cutoff", "ds", "h"], suffixes=("", "_b"))
    if len(df) != len(lstm):
        raise ValueError(f"fold misalignment: merged {len(df)} rows vs {len(lstm)} LSTM rows")
    if not np.allclose(df["y_true"], df["y"], rtol=1e-4, atol=1e-4):
        raise ValueError("shared actuals differ between the two CV files")
    return df


def pooled_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Table 5.1: pooled RMSE/MAE/WAPE/MAPE for all five models on identical folds."""
    rows = {"LSTM": M.all_metrics(df["y_true"], df["y_pred"])}
    for m in BASELINES:
        rows[m] = M.all_metrics(df["y_true"], df[m])
    return pd.DataFrame(rows).T[["rmse", "mae", "wape", "mape"]]


def per_step_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Table 5.2: per-forecast-step RMSE and WAPE, LSTM vs AutoARIMA."""
    out = []
    for h, g in df.groupby("h"):
        out.append({"h": int(h),
                    "rmse_LSTM": M.rmse(g["y_true"], g["y_pred"]),
                    "rmse_ARIMA": M.rmse(g["y_true"], g["AutoARIMA"]),
                    "wape_LSTM": M.wape(g["y_true"], g["y_pred"]),
                    "wape_ARIMA": M.wape(g["y_true"], g["AutoARIMA"])})
    return pd.DataFrame(out)


def per_series_rmse(df: pd.DataFrame) -> pd.DataFrame:
    """Observation 3: per-series RMSE for LSTM / AutoARIMA / per-series mean + win flags."""
    def series_rmse(col):
        return df.groupby("series_id").apply(
            lambda g: M.rmse(g["y_true"], g[col]), include_groups=False)
    t = pd.DataFrame({"LSTM": series_rmse("y_pred"),
                      "ARIMA": series_rmse("AutoARIMA"),
                      "MEAN": series_rmse("HistoricAverage")})
    t["LSTM_lt_ARIMA"] = t["LSTM"] < t["ARIMA"]      # LSTM RMSE lower than AutoARIMA
    t["LSTM_lt_MEAN"] = t["LSTM"] < t["MEAN"]        # LSTM RMSE lower than the per-series mean
    return t.reset_index()


def main():
    df = load_merged()
    n_series = df["series_id"].nunique()

    pooled = pooled_accuracy(df)
    by_step = per_step_accuracy(df)
    per_series = per_series_rmse(df)
    win_arima = int(per_series["LSTM_lt_ARIMA"].sum())
    win_mean = int(per_series["LSTM_lt_MEAN"].sum())

    pooled.round(3).to_csv(RESULTS / "report_pooled_accuracy.csv")
    by_step.round(3).to_csv(RESULTS / "report_per_step.csv", index=False)
    per_series.round(3).to_csv(RESULTS / "report_per_series_rmse.csv", index=False)

    print(f"[report] {n_series} series x {df['cutoff'].nunique()} origins x "
          f"{df['h'].nunique()} steps = {len(df)} paired points\n")
    print("[report] Table 5.1 - pooled accuracy (lower is better):")
    print(pooled.round(2).to_string(), "\n")
    print("[report] Table 5.2 - per forecast step (LSTM vs AutoARIMA):")
    print(by_step.round(2).to_string(index=False), "\n")
    print("[report] per-series RMSE (Observation 3):")
    print(per_series.round(1).to_string(index=False), "\n")
    print(f"[report] Observation 3: LSTM RMSE beats AutoARIMA on {win_arima}/{n_series} "
          f"series; beats the per-series mean on {win_mean}/{n_series}.")
    print("[report] wrote report_pooled_accuracy.csv, report_per_step.csv, "
          "report_per_series_rmse.csv")


if __name__ == "__main__":
    main()
