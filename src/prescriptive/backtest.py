"""
backtest.py - quantify prescriptive utility with a three-arm ablation (step 8).

Most forecasting work stops at RMSE. This replays the held-out test period week by week
and asks the operational question: do the prescriptions actually help, and WHICH
ingredient helps? Three policies share the SAME (s, S) reorder mechanics (via
prescribe()), the same realised demand and the same starting stock - they differ ONLY in
the demand estimate and/or the uncertainty that sizes the safety stock (brief s9):

  (a) naive     - demand = moving average of the last K weeks; sigma = std of those
                  weeks. The no-ML anchor.
  (b) lstm_hist - demand = the LSTM forecast mean; sigma = HISTORICAL demand std over
                  the last HIST_STD_WEEKS (the textbook safety stock).
  (c) lstm_mc   - demand = the LSTM forecast mean; sigma_LT = the model's own calibrated
                  uncertainty (MC path-sum std + residual term). The full system.

(c) vs (b) isolates the value of MODEL-PRODUCED uncertainty - the claimed contribution.
(b) vs (a) isolates the value of the forecast itself.

False alarms: the stockout-risk warning is advisory - it triggers no extra order beyond
the routine (s, S) reorder - so "false" can be judged directly against the realised
trajectory: an alert is false when realised lead-time demand never exceeded the stock
available. The counterfactual "had we not acted on the alert" IS the observed path.

Run from the repo root:  python -m src.prescriptive.backtest
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .rules import prescribe, PolicyParams
from ..config import RESULTS, STANDARD, INVENTORY_SIM, TEST_WEEKS, LEAD_TIME

LSTM_PRED = RESULTS / "lstm_cv_predictions.csv"

LEAD = LEAD_TIME     # replenishment lead time (weeks); from config (<= forecast horizon)
MA_WINDOW = 4        # K: weeks of history the naive policy averages over
HIST_STD_WEEKS = 13  # ~one quarter of history for the textbook (historical-sigma) safety stock
POLICIES = ["lstm_mc", "lstm_hist", "naive"]


def _levels(mu, sigma, sig_lt, inv, params):
    """Reuse the rule engine to get (reorder_point, order_up_to) for any demand estimate."""
    p = prescribe(mu, sigma, inv, params, sigma_lt=sig_lt)
    return p["reorder_point"], p["order_up_to"]


def simulate(actual, test_start, last_decision, init_inv, get_demand, params):
    """Forward inventory simulation over the test weeks for one SKU under one policy.

    get_demand(w) -> (mu[LEAD], sigma[LEAD], sigma_lt_or_None): the demand estimate made
    at the start of week w. Returns a per-week DataFrame and a metrics dict.
    """
    n = len(actual)
    inv = float(init_inv)
    pipeline: dict[int, float] = {}
    recs, alerts, false_alarms, warns = [], 0, 0, []

    for w in range(test_start, n):
        inv += pipeline.pop(w, 0.0)              # receive scheduled arrivals
        inv_start = inv
        alerted = False
        if w <= last_decision:
            mu, sigma, sig_lt = get_demand(w)
            rop, order_up_to = _levels(mu, sigma, sig_lt, inv, params)
            if inv <= rop:                         # routine (s, S) reorder -> drives inventory
                order = max(0.0, order_up_to - inv)
                if order > 0:
                    pipeline[w + LEAD] = pipeline.get(w + LEAD, 0.0) + order
            # Stockout-risk WARNING (advisory, separate from routine reordering): estimated
            # lead-time demand exceeds the stock available before replenishment can arrive.
            # True positive if a stockout really occurs in that window; false alarm otherwise.
            avail = inv_start + sum(pipeline.get(t, 0.0) for t in range(w, w + LEAD))
            if mu.sum() > avail:
                alerted = True
                alerts += 1
                cum = np.cumsum(actual[w:w + LEAD])
                if avail >= cum[-1]:               # predicted a stockout that did not happen
                    false_alarms += 1
                else:
                    warns.append(int(np.argmax(cum > avail)))   # weeks until the real stockout
        demand = actual[w]
        sold = min(demand, inv)
        inv -= sold
        recs.append({"w": w, "inv_start": inv_start, "demand": demand,
                     "sold": sold, "unmet": demand - sold, "alert": alerted})

    df = pd.DataFrame(recs)
    total_demand = df["demand"].sum()
    metrics = {
        "service_level": df["sold"].sum() / total_demand if total_demand else 1.0,
        "unmet_units": df["unmet"].sum(),
        "stockout_weeks": int((df["unmet"] > 0).sum()),
        "avg_inventory": df["inv_start"].mean(),
        "alerts": alerts,
        "false_alarm_rate": false_alarms / alerts if alerts else 0.0,
        "avg_warning_weeks": float(np.mean(warns)) if warns else float("nan"),
    }
    return df, metrics


def run():
    preds = pd.read_csv(LSTM_PRED, parse_dates=["cutoff", "ds"])
    std = pd.read_parquet(STANDARD)
    inv_df = pd.read_parquet(INVENTORY_SIM)
    weeks = np.sort(std["date"].unique())
    test_start = len(weeks) - TEST_WEEKS
    week_pos = {pd.Timestamp(w): i for i, w in enumerate(weeks)}
    # last week at which a forecast-based decision is possible (needs a forecast at cutoff w-1)
    last_decision = max(week_pos[pd.Timestamp(c)] for c in preds["cutoff"].unique()) + 1

    fc = {(s, week_pos[pd.Timestamp(c)] + 1): g.sort_values("h")
          for (s, c), g in preds.groupby(["series_id", "cutoff"])}
    params = PolicyParams()

    rows, traj = [], {}
    for sid in sorted(std["series_id"].unique()):
        actual = std[std["series_id"] == sid].sort_values("date")["sales"].to_numpy()
        init_inv = float(inv_df[(inv_df["series_id"] == sid) &
                                (inv_df["date"] == weeks[test_start])]["inventory"].iloc[0])

        def lstm_mc(w, sid=sid):
            g = fc[(sid, w)]
            return (g["y_pred"].to_numpy()[:LEAD], g["y_std"].to_numpy()[:LEAD],
                    float(g["y_std_cum"].to_numpy()[LEAD - 1]))

        def lstm_hist(w, sid=sid, actual=actual):
            g = fc[(sid, w)]
            sig = float(actual[max(0, w - HIST_STD_WEEKS):w].std())
            # sigma_lt=None -> prescribe() falls back to sqrt(sum sigma^2) = sigma*sqrt(L),
            # i.e. exactly the textbook historical-sigma safety stock.
            return g["y_pred"].to_numpy()[:LEAD], np.full(LEAD, sig), None

        def naive(w, actual=actual):
            recent = actual[w - MA_WINDOW:w]
            return np.full(LEAD, recent.mean()), np.full(LEAD, recent.std()), None

        fns = {"lstm_mc": lstm_mc, "lstm_hist": lstm_hist, "naive": naive}
        res = {}
        for pol in POLICIES:
            res[pol] = simulate(actual, test_start, last_decision, init_inv, fns[pol], params)
        rows.append({"series_id": sid, **{f"{k}_{pol}": v for pol in POLICIES
                                          for k, v in res[pol][1].items()}})
        traj[sid] = res

    per = pd.DataFrame(rows)
    summary = aggregate(per, std, weeks, test_start)
    figure(traj, weeks, test_start, per)
    per.round(3).to_csv(RESULTS / "backtest_per_sku.csv", index=False)
    summary.round(3).to_csv(RESULTS / "backtest_summary.csv", index=False)
    report(summary)


def aggregate(per, std, weeks, test_start):
    """Pool demand-weighted service level + totals across SKUs, per policy."""
    total_demand = std[std["date"].isin(weeks[test_start:])]["sales"].sum()
    out = []
    for pol in POLICIES:
        unmet = per[f"unmet_units_{pol}"].sum()
        out.append({
            "policy": pol,
            "service_level": (total_demand - unmet) / total_demand if total_demand else 1.0,
            "unmet_units": unmet,
            "stockout_weeks": per[f"stockout_weeks_{pol}"].sum(),
            "avg_inventory": per[f"avg_inventory_{pol}"].mean(),
            "alerts": per[f"alerts_{pol}"].sum(),
            "false_alarm_rate": per[f"false_alarm_rate_{pol}"].mean(),
            "avg_warning_weeks": per[f"avg_warning_weeks_{pol}"].mean(),
        })
    return pd.DataFrame(out)


def figure(traj, weeks, test_start, per):
    """Inventory trajectory under the full system vs naive, for the SKU where they differ most."""
    diff = per["unmet_units_naive"] - per["unmet_units_lstm_mc"]
    sid = per.loc[diff.idxmax(), "series_id"]
    df_f = traj[sid]["lstm_mc"][0]
    df_n = traj[sid]["naive"][0]
    wk = weeks[test_start: test_start + len(df_f)]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(wk, df_f["inv_start"], color="#2a9d4a", lw=1.8, label="inventory - lstm_mc (full system)")
    ax.plot(wk, df_n["inv_start"], color="#d5573a", lw=1.8, ls="--", label="inventory - naive policy")
    ax.bar(wk, df_f["demand"], width=4, color="#3a7bd5", alpha=0.35, label="actual demand")
    so = df_n[df_n["unmet"] > 0]
    ax.scatter(weeks[so["w"].to_numpy()], np.zeros(len(so)),
               color="#d5573a", marker="v", s=60, zorder=5, label="naive stockout")
    ax.set_title(f"Backtest - SKU {sid}: the forecast-driven policy holds stock through demand "
                 f"the naive one misses")
    ax.set_xlabel("Week")
    ax.set_ylabel("Units")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_backtest_example.png", dpi=120)
    plt.close(fig)


def report(summary):
    print("[backtest] three-arm ablation over the held-out period "
          "(same rules, same realised demand, same starting stock):\n")
    print(summary.round(3).to_string(index=False))
    c = summary.set_index("policy")
    print(f"\n[backtest] value of the FORECAST ((b) lstm_hist vs (a) naive): "
          f"unmet {c.loc['lstm_hist', 'unmet_units']:.0f} vs {c.loc['naive', 'unmet_units']:.0f} units | "
          f"service {c.loc['lstm_hist', 'service_level']:.1%} vs {c.loc['naive', 'service_level']:.1%}")
    print(f"[backtest] value of MODEL UNCERTAINTY ((c) lstm_mc vs (b) lstm_hist): "
          f"unmet {c.loc['lstm_mc', 'unmet_units']:.0f} vs {c.loc['lstm_hist', 'unmet_units']:.0f} units | "
          f"avg stock {c.loc['lstm_mc', 'avg_inventory']:.0f} vs {c.loc['lstm_hist', 'avg_inventory']:.0f} | "
          f"false alarms {c.loc['lstm_mc', 'false_alarm_rate']:.1%} vs {c.loc['lstm_hist', 'false_alarm_rate']:.1%}")
    print(f"[backtest] full system: {c.loc['lstm_mc', 'alerts']:.0f} stockout-risk warnings, "
          f"false-alarm rate {c.loc['lstm_mc', 'false_alarm_rate']:.1%}, "
          f"avg warning lead {c.loc['lstm_mc', 'avg_warning_weeks']:.1f} wks")
    print("\n[backtest] wrote results/backtest_summary.csv, backtest_per_sku.csv, "
          "fig_backtest_example.png")


if __name__ == "__main__":
    run()
