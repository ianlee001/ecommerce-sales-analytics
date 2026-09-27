"""
baselines.py - ARIMA + naive baselines on ONE shared rolling-origin protocol (step 3).

The LSTM is *evaluated against* these; we never claim it is guaranteed to beat ARIMA.
Models (all statsforecast - the same Hyndman-Khandakar auto.arima as pmdarima, but
maintained and fast on current Python):

  * AutoARIMA(season_length=1) - the headline comparison. Non-seasonal ON PURPOSE:
      annual (52-week) seasonality needs >= 2 full cycles to estimate but the training
      span is ~1.5 years, and a 4-week-ahead forecast does not need the 52-week cycle.
      Learning the seasonal shape from limited cycles is what the global LSTM (which
      gets week-of-year features) is meant to do better - that is the empirical
      question. (ARIMAX with Fourier terms is a possible extension, noted in the thesis.)
  * HistoricAverage - per-series mean of all history up to the origin. THE decisive
      sanity check: if the LSTM cannot beat this, the modelling adds nothing.
  * Naive - last observed value carried forward.
  * SeasonalNaive(52) - same week last year: the pure-seasonality yardstick.

Evaluation = rolling-origin cross-validation over the held-out TEST_WEEKS (config.py):
at each weekly origin, fit on everything up to that week and forecast HORIZON weeks
ahead. Test data is never seen during fitting (hard rule).

Run from the repo root:  python -m src.predictive.baselines
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, HistoricAverage, Naive, SeasonalNaive

from ..config import STANDARD, RESULTS, FREQ, HORIZON, TEST_WEEKS, CV_STEP, N_WINDOWS
from . import metrics as M

MODELS = ["AutoARIMA", "HistoricAverage", "Naive", "SeasonalNaive"]


def load_panel() -> pd.DataFrame:
    """Standard format -> statsforecast long panel [unique_id, ds, y]."""
    df = pd.read_parquet(STANDARD).rename(
        columns={"series_id": "unique_id", "date": "ds", "sales": "y"})
    return df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)


def run_cv(panel: pd.DataFrame) -> pd.DataFrame:
    """Rolling-origin CV for all four models on identical folds."""
    # n_jobs=1: in-process, deterministic; 20 short series is fast anyway.
    sf = StatsForecast(models=[AutoARIMA(season_length=1), HistoricAverage(), Naive(),
                               SeasonalNaive(season_length=52)], freq=FREQ, n_jobs=1)
    cv = sf.cross_validation(df=panel, h=HORIZON, step_size=CV_STEP, n_windows=N_WINDOWS)
    if "unique_id" not in cv.columns:          # some versions return it as the index
        cv = cv.reset_index()
    # forecast step h = weeks between the origin (cutoff) and the forecast date
    cv["h"] = ((cv["ds"] - cv["cutoff"]).dt.days // 7).astype(int)
    return cv


def summarise(cv: pd.DataFrame):
    """Pooled metrics per model + per-series and per-horizon detail for AutoARIMA."""
    pooled = pd.DataFrame({m: M.all_metrics(cv["y"], cv[m]) for m in MODELS}).T
    per_arima = (cv.groupby("unique_id")
                   .apply(lambda g: pd.Series(M.all_metrics(g["y"], g["AutoARIMA"])),
                          include_groups=False)
                   .reset_index().rename(columns={"unique_id": "series_id"}))
    by_h_arima = (cv.groupby("h")
                    .apply(lambda g: pd.Series(M.all_metrics(g["y"], g["AutoARIMA"])),
                           include_groups=False)
                    .reset_index())
    return pooled, per_arima, by_h_arima


def example_figure(panel: pd.DataFrame, cv: pd.DataFrame, sid: str):
    """Actual vs the rolling 1-week-ahead ARIMA forecast over the test period, one series."""
    hist = panel[panel["unique_id"] == sid]
    one = cv[(cv["unique_id"] == sid) & (cv["h"] == 1)].sort_values("ds")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hist["ds"], hist["y"], color="#3a7bd5", lw=1.2, label="actual")
    ax.plot(one["ds"], one["AutoARIMA"], color="#d5573a", lw=1.6, ls="--",
            label="ARIMA 1-week-ahead (rolling)")
    ax.axvspan(one["ds"].min(), one["ds"].max(), color="#cccccc", alpha=0.25,
               label="test period")
    ax.set_title(f"ARIMA baseline - SKU {sid}: actual vs rolling 1-week-ahead forecast")
    ax.set_xlabel("Week")
    ax.set_ylabel("Units")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_arima_example.png", dpi=120)
    plt.close(fig)


def main():
    panel = load_panel()
    print(f"[baselines] {panel['unique_id'].nunique()} series, "
          f"{panel['ds'].nunique()} weeks; holding out last {TEST_WEEKS} "
          f"(rolling origin: {N_WINDOWS} windows x H={HORIZON})")
    cv = run_cv(panel)
    pooled, per_arima, by_h_arima = summarise(cv)

    cv.to_csv(RESULTS / "baseline_cv_predictions.csv", index=False)
    pooled.round(3).to_csv(RESULTS / "baseline_metrics_pooled.csv")
    per_arima.round(3).to_csv(RESULTS / "baseline_arima_per_series.csv", index=False)
    example_figure(panel, cv, per_arima.sort_values("rmse").iloc[len(per_arima) // 2]["series_id"])

    print("\n[baselines] pooled over the held-out test period (lower is better):")
    print(pooled.round(2).to_string())
    print("\n[baselines] AutoARIMA error by forecast step (weeks ahead):")
    print(by_h_arima.round(2).to_string(index=False))
    print("\n[baselines] AutoARIMA per series (rounded):")
    print(per_arima.round(2).to_string(index=False))
    print("\n[baselines] wrote baseline_cv_predictions.csv, baseline_metrics_pooled.csv, "
          "baseline_arima_per_series.csv, fig_arima_example.png")


if __name__ == "__main__":
    main()
