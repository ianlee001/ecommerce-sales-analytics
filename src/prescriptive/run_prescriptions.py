"""
run_prescriptions.py - drive the rule engine end-to-end (build-order step 6 demo).

At the most recent decision origin in the held-out test period, for every SKU it takes
the LSTM's MC-Dropout forecast over the LEAD_TIME (mean + the path-sum lead-time std)
and the simulated current inventory, and produces a concrete, confidence-aware
prescription. This is the "recommended actions" panel the dashboard will surface, and
the same call the backtest replays at every week.

Run from the repo root:  python -m src.prescriptive.run_prescriptions
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .rules import prescribe, PolicyParams
from ..config import RESULTS, INVENTORY_SIM, LEAD_TIME

LSTM_PRED = RESULTS / "lstm_cv_predictions.csv"

ACTION_COLOR = {"RESTOCK": "#d5573a", "REVIEW_RESTOCK": "#e8a33d", "ON_TRACK": "#2a9d4a",
                "DISCOUNT": "#3a7bd5", "REVIEW_DISCOUNT": "#7d6fb0"}


def build(params: PolicyParams | None = None) -> pd.DataFrame:
    preds = pd.read_csv(LSTM_PRED, parse_dates=["cutoff", "ds"])
    inv_df = pd.read_parquet(INVENTORY_SIM)
    origin = preds["cutoff"].max()                       # most recent decision point
    snap = preds[preds["cutoff"] == origin]

    rows = []
    for sid, g in snap.groupby("series_id"):
        g = g.sort_values("h")
        first_week = g.loc[g["h"] == 1, "ds"].iloc[0]    # start of the forecast horizon
        inv = float(inv_df.loc[(inv_df["series_id"] == sid) &
                               (inv_df["date"] == first_week), "inventory"].iloc[0])
        # reorder decisions cover the LEAD_TIME, not the whole forecast horizon;
        # sigma_lt = the MC path-sum std of the LEAD_TIME-week total (no independence assumption)
        pres = prescribe(g["y_pred"].values[:LEAD_TIME], g["y_std"].values[:LEAD_TIME], inv,
                         params, sigma_lt=float(g["y_std_cum"].values[LEAD_TIME - 1]))
        rows.append({"series_id": sid, "origin": origin.date(), **pres})
    return pd.DataFrame(rows)


def figure(df: pd.DataFrame):
    """Current inventory vs reorder point, coloured by action (below the diagonal = restock)."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for action, gg in df.groupby("action"):
        ax.scatter(gg["reorder_point"], gg["current_inventory"], s=70,
                   color=ACTION_COLOR.get(action, "#777"), label=action,
                   edgecolor="white", zorder=3)
    lim = max(df["reorder_point"].max(), df["current_inventory"].max()) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color="#999", lw=1, label="inventory = reorder point")
    for _, r in df.iterrows():
        ax.annotate(r["series_id"], (r["reorder_point"], r["current_inventory"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Reorder point (forecast lead-time demand + safety stock)")
    ax.set_ylabel("Current inventory")
    ax.set_title("Prescriptions at the latest origin - points below the line need restocking")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_prescriptions.png", dpi=120)
    plt.close(fig)


def main():
    df = build()
    cols = ["series_id", "current_inventory", "lead_time_demand", "lead_time_demand_std",
            "reorder_point", "order_up_to", "weeks_of_cover_opt", "cv", "confidence",
            "action", "suggested_order_qty"]
    df[cols + ["message"]].to_csv(RESULTS / "prescriptions.csv", index=False)
    figure(df)

    origin = df["origin"].iloc[0]
    print(f"[rules] prescriptions at origin {origin} "
          f"(lead time = {LEAD_TIME} weeks, sigma_LT from MC path sums):\n")
    print(df[cols].to_string(index=False))
    print("\n[rules] action summary:")
    print(df["action"].value_counts().to_string())
    print("\n[rules] examples:")
    for _, r in df.head(3).iterrows():
        print(f"  - {r['series_id']} [{r['action']}, conf={r['confidence']}]: {r['message']}")
    print("\n[rules] wrote results/prescriptions.csv, fig_prescriptions.png")


if __name__ == "__main__":
    main()
