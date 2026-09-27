"""
eda_aggregates.py - business-level aggregates for the dashboard landing page (offline).

The modelling contract (standard format) deliberately carries only
series_id/date/sales/price - but the landing page should read like a shop dashboard
(revenue, orders, customers, markets). Those business numbers come from the CLEANED raw
transactions, are computed OFFLINE here (dataset-specific -> lives in src/data beside
the adapter, and reuses the adapter's own load/clean code), and are served read-only by
the API from one small JSON artifact.

Run from the repo root:  python -m src.data.eda_aggregates
"""
import json

import pandas as pd

from ..config import PROCESSED, RESULTS
from .adapter import clean, load_raw, to_weekly, week_grid

OUT = PROCESSED / "eda_aggregates.json"
TOP_MARKETS = 7
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def main():
    df = clean(load_raw())
    skus = pd.read_csv(RESULTS / "selected_skus.csv",
                       dtype={"series_id": str})["series_id"].tolist()

    # headline cards
    by_country = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
    totals = {
        "revenue": round(float(df["Revenue"].sum()), 2),
        "units": int(df["Quantity"].sum()),
        "invoices": int(df["Invoice"].nunique()),
        "products": int(df["StockCode"].nunique()),
        "customers": int(df["Customer ID"].nunique()),   # identified customers only
        "main_market": str(by_country.index[0]),
    }

    # revenue by market (pie): top N + Other
    markets = [{"name": str(c), "revenue": round(float(v), 2)}
               for c, v in by_country.head(TOP_MARKETS).items()]
    other = float(by_country.iloc[TOP_MARKETS:].sum())
    if other > 0:
        markets.append({"name": "Other", "revenue": round(other, 2)})

    # best months (horizontal bars): revenue by calendar month across the whole span
    month_rev = df.groupby(df["InvoiceDate"].dt.month)["Revenue"].sum()
    best_months = [{"month": MONTHS[int(m) - 1], "revenue": round(float(v), 2)}
                   for m, v in month_rev.items()]

    # whole-shop weekly unit totals (same trimmed weekly grid as the adapter) - the
    # Overview chart must match the headline cards' whole-shop scope, not just Top-N
    weekly = to_weekly(df)
    grid = week_grid(weekly)
    totals_by_week = (weekly.groupby("week")["sales"].sum()
                      .reindex(grid, fill_value=0))
    total_weekly = [{"date": str(d.date()), "sales": int(v)}
                    for d, v in totals_by_week.items()]

    # product table rows for the modelled Top-N (description from the raw data)
    sub = df[df["StockCode"].isin(skus)]
    desc = (sub[sub["Description"].str.lower() != "nan"]
            .groupby("StockCode")["Description"]
            .agg(lambda s: s.value_counts().index[0]))
    agg = sub.groupby("StockCode").agg(units=("Quantity", "sum"),
                                       revenue=("Revenue", "sum"),
                                       avg_price=("Price", "mean"))
    product_table = [{
        "series_id": s,
        "description": str(desc.get(s, "")).title(),
        "units": int(agg.loc[s, "units"]),
        "revenue": round(float(agg.loc[s, "revenue"]), 2),
        "avg_price": round(float(agg.loc[s, "avg_price"]), 2),
    } for s in skus]

    OUT.write_text(json.dumps({"totals": totals, "markets": markets,
                               "best_months": best_months,
                               "total_weekly": total_weekly,
                               "product_table": product_table}, indent=2),
                   encoding="utf-8")
    print(f"[eda-agg] totals: {totals}")
    print(f"[eda-agg] markets: {len(markets)} rows | months: {len(best_months)} "
          f"| products: {len(product_table)}")
    print(f"[eda-agg] wrote {OUT}")


if __name__ == "__main__":
    main()
