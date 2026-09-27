"""
elasticity.py - estimate the price elasticity epsilon for the discount what-if (step 5).

The dashboard's discount slider moves demand via Q' = Q * (P'/P)^epsilon (brief s7). The
LSTM does NOT learn price response (price is not a model input; the real price->demand
signal in this data is weak), so epsilon is a DOCUMENTED ASSUMPTION: estimated from the
data where that estimate is reliable, otherwise a literature value with a sensitivity
range.

Estimation: per-series OLS of log(sales) ~ log(price) on TRAINING-period weeks with
sales > 0 (a log-log slope IS the elasticity). "Reliable" is judged by criteria FIXED IN
ADVANCE (brief s7) - decided before looking at any results, so we cannot cherry-pick:

    R1  n >= 30 usable weeks            (enough observations)
    R2  price CV >= 0.05                (the price actually moved; a flat price
                                         identifies nothing)
    R3  slope < 0                       (economically sane: price up -> demand down)
    R4  95% CI excludes 0               (statistically distinguishable from no effect)

Honest caveat (thesis): weekly mean price also shifts with the retail/wholesale customer
mix and with seasonal assortment, so even a "reliable" estimate is correlational, not
causal - hence the fallback and the sensitivity range, and hence phrasing the slider as
a what-if, never a guarantee.

Fallback: epsilon = -1.5 (typical retail literature value), sensitivity [-2.5, -0.5].

Outputs:
    results/elasticity.csv           per-series estimates + criteria flags (thesis table)
    data/processed/elasticity.json   what the Flask API serves the what-if slider from

Run from the repo root:  python -m src.prescriptive.elasticity
"""
import json

import numpy as np
import pandas as pd

from ..config import STANDARD, PROCESSED, RESULTS, TEST_WEEKS

# Pre-specified reliability criteria (fixed before estimation - see module docstring)
MIN_OBS = 30
MIN_PRICE_CV = 0.05
Z95 = 1.96

# Fallback assumption (literature) + sensitivity range for the thesis discussion
DEFAULT_EPS = -1.5
SENSITIVITY = (-2.5, -0.5)


def _ols_loglog(price: np.ndarray, sales: np.ndarray):
    """OLS slope of log(sales) on log(price): returns (slope, half-width of 95% CI)."""
    x = np.log(price)
    y = np.log(sales)
    xc = x - x.mean()
    sxx = float((xc ** 2).sum())
    slope = float((xc * (y - y.mean())).sum() / sxx)
    resid = y - (y.mean() + slope * xc)
    dof = len(x) - 2
    se = float(np.sqrt((resid ** 2).sum() / dof / sxx))
    return slope, Z95 * se


def estimate() -> pd.DataFrame:
    df = pd.read_parquet(STANDARD)
    weeks = np.sort(df["date"].unique())
    train_weeks = set(weeks[:-TEST_WEEKS])          # training period only (hard rule)
    train = df[df["date"].isin(train_weeks)]

    rows = []
    for sid, g in train.groupby("series_id"):
        g = g[g["sales"] > 0]                       # log() needs positive sales
        n = len(g)
        price_cv = float(g["price"].std() / g["price"].mean()) if n else 0.0
        eps = ci = np.nan
        if n >= 3 and g["price"].nunique() > 1:
            eps, ci = _ols_loglog(g["price"].to_numpy(), g["sales"].to_numpy())
        checks = {
            "R1_enough_obs": n >= MIN_OBS,
            "R2_price_moved": price_cv >= MIN_PRICE_CV,
            "R3_negative": bool(eps < 0) if np.isfinite(eps) else False,
            "R4_ci_excl_0": bool(abs(eps) > ci) if np.isfinite(eps) else False,
        }
        rows.append({"series_id": sid, "n_weeks": n, "price_cv": round(price_cv, 3),
                     "epsilon": round(eps, 3) if np.isfinite(eps) else np.nan,
                     "ci95_halfwidth": round(ci, 3) if np.isfinite(ci) else np.nan,
                     **checks, "reliable": all(checks.values())})
    return pd.DataFrame(rows).sort_values("series_id").reset_index(drop=True)


def main():
    tab = estimate()
    tab.to_csv(RESULTS / "elasticity.csv", index=False)

    per_series = {r.series_id: round(float(r.epsilon), 3)
                  for r in tab.itertuples() if r.reliable}
    artifact = {
        "default": DEFAULT_EPS,
        "sensitivity": list(SENSITIVITY),
        "criteria": {"min_obs": MIN_OBS, "min_price_cv": MIN_PRICE_CV,
                     "negative_sign": True, "ci95_excludes_0": True},
        "per_series": per_series,       # only series whose own estimate passed all criteria
    }
    (PROCESSED / "elasticity.json").write_text(json.dumps(artifact, indent=2),
                                               encoding="utf-8")

    n_rel = int(tab["reliable"].sum())
    print(f"[elasticity] per-series log(sales)~log(price) OLS on the TRAIN period "
          f"({len(tab)} series):\n")
    print(tab.to_string(index=False))
    print(f"\n[elasticity] reliable estimates: {n_rel}/{len(tab)} "
          f"(criteria: n>={MIN_OBS}, price CV>={MIN_PRICE_CV}, negative, 95% CI excl. 0)")
    if n_rel:
        rel = tab[tab["reliable"]]
        print(f"[elasticity] reliable epsilon range: {rel['epsilon'].min():.2f} .. "
              f"{rel['epsilon'].max():.2f} (median {rel['epsilon'].median():.2f})")
    print(f"[elasticity] fallback for all other series: {DEFAULT_EPS} "
          f"(sensitivity {SENSITIVITY[0]} .. {SENSITIVITY[1]})")
    print("[elasticity] wrote results/elasticity.csv, data/processed/elasticity.json")


if __name__ == "__main__":
    main()
