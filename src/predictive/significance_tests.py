"""Paired significance tests for forecast accuracy comparisons.

Answers one question with evidence instead of rhetoric: is the pooled RMSE gap
between the LSTM and each baseline statistically real, or within sampling noise?

Two complementary procedures, both operating on the per-window predictions saved
by the offline pipeline (results/lstm_cv_predictions.csv and
results/baseline_cv_predictions.csv), so no model is re-run here:

1. Diebold-Mariano (DM) tests per series, on the loss differential
   d_t = loss(LSTM) - loss(baseline), collapsed to one value per rolling origin
   (mean over the H=4 horizon steps). Consecutive origins overlap by H-1 weeks,
   so the differential is autocorrelated; the DM variance uses a Newey-West
   (Bartlett) estimator with lag H-1, and the small-sample correction of
   Harvey, Leybourne & Newbold (1997) is applied with a t(n-1) reference.
   Both squared-error and absolute-error losses are tested, because the pooled
   RMSE and WAPE rankings disagree and each loss speaks to one of them.

2. A series-level cluster bootstrap for the POOLED metrics: resample the 20
   series with replacement (keeping each series' errors intact, which preserves
   within-series autocorrelation), recompute pooled RMSE / WAPE for both models,
   and report the 95% percentile CI of the difference. If the CI spans zero the
   pooled gap is not distinguishable from sampling noise at the 5% level.

Outputs:
    results/significance_dm_per_series.csv   one row per (series, baseline, loss)
    results/significance_pooled_bootstrap.csv one row per (baseline, metric)
    and a console summary used for the thesis text.

Run from the repo root:  python -m src.predictive.significance_tests
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path("results")
HORIZON = 4              # forecast steps per origin (weeks)
NW_LAG = HORIZON - 1     # Newey-West truncation: adjacent origins overlap by H-1
N_BOOT = 10_000
SEED = 42
BASELINES = ["AutoARIMA", "HistoricAverage", "Naive", "SeasonalNaive"]


def load_paired_predictions() -> pd.DataFrame:
    """Merge LSTM and baseline predictions on identical forecast targets."""
    lstm = pd.read_csv(RESULTS_DIR / "lstm_cv_predictions.csv",
                       parse_dates=["cutoff", "ds"])
    base = pd.read_csv(RESULTS_DIR / "baseline_cv_predictions.csv",
                       parse_dates=["cutoff", "ds"])
    base = base.rename(columns={"unique_id": "series_id"})
    merged = lstm.merge(
        base[["series_id", "cutoff", "ds", "y"] + BASELINES],
        on=["series_id", "cutoff", "ds"], how="inner", validate="one_to_one")
    # The two files must describe the same targets, or the pairing is invalid.
    if not np.allclose(merged["y_true"], merged["y"]):
        raise ValueError("y_true mismatch between LSTM and baseline files")
    return merged


def newey_west_dm(d: np.ndarray, h: int) -> tuple[float, float]:
    """DM statistic with NW(lag=h-1) variance and HLN correction; t(n-1) p-value."""
    n = len(d)
    dbar = d.mean()
    u = d - dbar
    gamma0 = (u @ u) / n
    var = gamma0
    for k in range(1, min(h - 1, n - 1) + 1):
        w = 1.0 - k / h                       # Bartlett weight, truncation h-1
        gamma_k = (u[:-k] @ u[k:]) / n
        var += 2.0 * w * gamma_k
    var = max(var, 1e-12)                     # guard: NW variance can go negative
    dm = dbar / np.sqrt(var / n)
    # Harvey-Leybourne-Newbold small-sample correction for h-step forecasts
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return dm_hln, p


def dm_per_series(df: pd.DataFrame) -> pd.DataFrame:
    """DM test per (series, baseline, loss); differential collapsed per origin."""
    rows = []
    e_lstm = df["y_pred"] - df["y_true"]
    for baseline in BASELINES:
        e_base = df[baseline] - df["y_true"]
        for loss_name, loss in [("squared", lambda e: e ** 2),
                                ("absolute", lambda e: e.abs())]:
            diff = pd.DataFrame({
                "series_id": df["series_id"], "cutoff": df["cutoff"],
                "d": loss(e_lstm) - loss(e_base)})
            per_origin = (diff.groupby(["series_id", "cutoff"])["d"]
                          .mean().reset_index().sort_values("cutoff"))
            for sid, grp in per_origin.groupby("series_id"):
                dm, p = newey_west_dm(grp["d"].to_numpy(), HORIZON)
                rows.append({"series_id": sid, "baseline": baseline,
                             "loss": loss_name, "n_origins": len(grp),
                             "dm_stat": dm, "p_value": p,
                             "favours": ("LSTM" if dm < 0 else baseline)
                                        if p < 0.05 else "n.s."})
    return pd.DataFrame(rows)


def pooled_bootstrap(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Series-level cluster bootstrap CI for pooled RMSE / WAPE differences."""
    rows = []
    series_ids = df["series_id"].unique()
    groups = {sid: g for sid, g in df.groupby("series_id")}

    def pooled_metrics(frame: pd.DataFrame, col: str) -> tuple[float, float]:
        err = frame[col] - frame["y_true"]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        wape = float(err.abs().sum() / frame["y_true"].abs().sum())
        return rmse, wape

    for baseline in BASELINES:
        point = {}
        for metric_idx, metric in [(0, "RMSE"), (1, "WAPE")]:
            point[metric] = (pooled_metrics(df, "y_pred")[metric_idx]
                             - pooled_metrics(df, baseline)[metric_idx])
        draws = {"RMSE": np.empty(N_BOOT), "WAPE": np.empty(N_BOOT)}
        for b in range(N_BOOT):
            sample = pd.concat(
                [groups[sid] for sid in rng.choice(series_ids, len(series_ids))],
                ignore_index=True)
            for metric_idx, metric in [(0, "RMSE"), (1, "WAPE")]:
                draws[metric][b] = (pooled_metrics(sample, "y_pred")[metric_idx]
                                    - pooled_metrics(sample, baseline)[metric_idx])
        for metric in ["RMSE", "WAPE"]:
            lo, hi = np.percentile(draws[metric], [2.5, 97.5])
            share_pos = float(np.mean(draws[metric] > 0))
            p_boot = 2.0 * min(share_pos, 1.0 - share_pos)
            rows.append({"baseline": baseline, "metric": metric,
                         "delta_LSTM_minus_baseline": point[metric],
                         "ci95_low": lo, "ci95_high": hi,
                         "p_bootstrap": p_boot,
                         "significant_5pct": not (lo <= 0.0 <= hi)})
    return pd.DataFrame(rows)


def main() -> None:
    df = load_paired_predictions()
    n_series = df["series_id"].nunique()
    n_origins = df["cutoff"].nunique()
    print(f"paired rows: {len(df)}  ({n_series} series x {n_origins} origins "
          f"x H={HORIZON})")

    # sanity: reproduce the headline pooled numbers before testing anything
    for col, label in [("y_pred", "LSTM")] + [(b, b) for b in BASELINES]:
        err = df[col] - df["y_true"]
        print(f"pooled RMSE {label:16s} {np.sqrt(np.mean(err ** 2)):7.1f}   "
              f"WAPE {err.abs().sum() / df['y_true'].abs().sum():6.3f}")

    dm = dm_per_series(df)
    dm.to_csv(RESULTS_DIR / "significance_dm_per_series.csv", index=False)
    print("\nDM per series (p<0.05 counts), LSTM vs baseline:")
    for baseline in BASELINES:
        for loss in ["squared", "absolute"]:
            sub = dm[(dm["baseline"] == baseline) & (dm["loss"] == loss)]
            n_lstm = (sub["favours"] == "LSTM").sum()
            n_base = (sub["favours"] == baseline).sum()
            n_ns = (sub["favours"] == "n.s.").sum()
            print(f"  vs {baseline:16s} [{loss:8s}]  LSTM better: {n_lstm:2d}  "
                  f"{baseline} better: {n_base:2d}  n.s.: {n_ns:2d}")

    rng = np.random.default_rng(SEED)
    boot = pooled_bootstrap(df, rng)
    boot.to_csv(RESULTS_DIR / "significance_pooled_bootstrap.csv", index=False)
    print("\nPooled differences (LSTM - baseline), series-cluster bootstrap 95% CI:")
    for _, r in boot.iterrows():
        sig = "SIGNIFICANT" if r["significant_5pct"] else "not significant"
        print(f"  vs {r['baseline']:16s} d{r['metric']:5s} "
              f"{r['delta_LSTM_minus_baseline']:+8.2f}  "
              f"CI [{r['ci95_low']:+8.2f}, {r['ci95_high']:+8.2f}]  "
              f"p~{r['p_bootstrap']:.3f}  -> {sig}")


if __name__ == "__main__":
    main()
