"""
adapter.py - Online Retail II -> standard internal format.

This is the ONLY dataset-specific component. Everything above it (LSTM, rule engine,
dashboard) consumes ONLY the standard format produced here, so the data source is
swappable: a different dataset just needs a different adapter.

Pipeline:
    raw transactions -> dedupe + clean -> weekly per-SKU grid (sales + price)
    -> trim partial edge weeks -> select Top-N series on the TRAINING PERIOD ONLY
    (by revenue, density-filtered) -> standard format (+ separate simulated inventory).

Standard internal format (the contract - see CLAUDE.md): one row per (series_id, date):
    series_id  (StockCode)
    date       (week-start Monday)
    sales      (units sold that week; the forecast target)
    price      (mean unit price that week; kept for the elasticity what-if, NOT a model input)

Inventory is NOT part of the contract (Online Retail II has no stock data): the simulated
trajectory - a documented assumption - is written to a SEPARATE file, inventory_sim.parquet.

Run from the repo root:  python -m src.data.adapter
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import (RAW, PROCESSED, STANDARD, INVENTORY_SIM, RESULTS,
                      TEST_WEEKS, TOP_N, ADI_MAX, LEAD_TIME)
from .inventory import simulate_inventory

CACHE = PROCESSED / "eda_cache_combined.parquet"   # parquet read-cache (Excel is slow)

# Documented cleaning / simulation parameters (justify in the thesis)
PRODUCT_CODE = r"^\d{5}[A-Za-z]{0,2}$"  # a "real" product StockCode: 5 digits + up to 2
                                        # letters (colour/size variants like 15056BL).
                                        # Drops POST, M, DOT, bank charges, gift_* ...
                                        # (exactly what it drops is verified in the EDA cell).
REVIEW_WEEKS = 2        # periodic-review window for the order-up-to level
SERVICE_Z = 1.65        # ~95% service level for the BASELINE inventory simulation


# --------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Load both Excel sheets (2009-2010, 2010-2011) via a parquet cache."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    xl = pd.ExcelFile(RAW)
    df = pd.concat([xl.parse(s) for s in xl.sheet_names], ignore_index=True)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Authoritative, documented cleaning (see brief s4.1)."""
    df = df.copy()
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    n0 = len(df)
    df = df.drop_duplicates()                              # 1. the two sheets overlap around
    #    Dec 2010 and re-list the same invoices; there are also doubled entry lines.
    print(f"[clean] dropped {n0 - len(df):,} duplicate rows (sheet overlap + double entries)")
    df = df[~df["Invoice"].str.startswith("C")]            # 2. cancellations
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]      # 3. returns / adjustments / freebies
    df = df[df["StockCode"].str.match(PRODUCT_CODE)]       # 4. real product codes only
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["Price"]
    # Missing Customer ID is kept - it does not affect sales aggregation.
    return df


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction lines to (StockCode, week-start Monday) weekly totals."""
    week = (df["InvoiceDate"] - pd.to_timedelta(df["InvoiceDate"].dt.dayofweek, unit="D")).dt.normalize()
    df = df.assign(week=week)
    return (df.groupby(["StockCode", "week"])
              .agg(sales=("Quantity", "sum"), revenue=("Revenue", "sum"),
                   price=("Price", "mean"))
              .reset_index())


def week_grid(weekly: pd.DataFrame) -> pd.DatetimeIndex:
    """All Mondays spanned by the data, with the PARTIAL first and last buckets trimmed:
    the data starts Tue 2009-12-01 and ends Fri 2011-12-09, so the edge buckets cover
    fewer than 7 days and would read as artificially low demand - the last one right
    inside the test period."""
    weeks = pd.date_range(weekly["week"].min(), weekly["week"].max(), freq="7D")
    return weeks[1:-1]


def select_series(weekly: pd.DataFrame, weeks: pd.DatetimeIndex):
    """Top-N SKUs by revenue computed on the TRAINING WEEKS ONLY (hard rule: selecting on
    full-period revenue would peek at test-period popularity), keeping only series dense
    enough to forecast weekly (ADI = average demand interval <= ADI_MAX)."""
    train_weeks = weeks[:-TEST_WEEKS]
    train = weekly[weekly["week"].isin(train_weeks)]
    revenue = train.groupby("StockCode")["revenue"].sum()
    active = train.groupby("StockCode")["week"].nunique()
    adi = len(train_weeks) / active                        # 1.0 = sells every train week
    eligible = adi[adi <= ADI_MAX].index                   # drop intermittent / single-bulk SKUs
    chosen = revenue.loc[eligible].sort_values(ascending=False).head(TOP_N).index.tolist()
    return chosen, revenue, adi


def build():
    """Run the full pipeline; write the standard format, inventory file and metadata."""
    df = clean(load_raw())
    weekly = to_weekly(df)
    weeks = week_grid(weekly)
    chosen, revenue, adi = select_series(weekly, weeks)

    frames, inv_frames, inv_params = [], [], {}
    for sku in chosen:
        g = weekly[weekly["StockCode"] == sku].set_index("week")
        sales = g["sales"].reindex(weeks, fill_value=0).astype(float)     # gap weeks: 0 sales
        price = g["price"].reindex(weeks).ffill().bfill().astype(float)   # gap weeks: carry price
        frames.append(pd.DataFrame({"series_id": sku, "date": weeks,
                                    "sales": sales.values, "price": price.values.round(2)}))
        inventory, params = simulate_inventory(sales.values, LEAD_TIME, REVIEW_WEEKS, SERVICE_Z)
        inv_params[sku] = params
        inv_frames.append(pd.DataFrame({"series_id": sku, "date": weeks, "inventory": inventory}))

    std = pd.concat(frames, ignore_index=True)
    inv = pd.concat(inv_frames, ignore_index=True)

    RESULTS.mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    std.to_parquet(STANDARD)
    std.to_csv(STANDARD.with_suffix(".csv"), index=False)   # human-readable copy
    inv.to_parquet(INVENTORY_SIM)
    meta = pd.DataFrame({"series_id": chosen})
    meta["train_revenue"] = meta["series_id"].map(revenue).round(2)
    meta["train_adi"] = meta["series_id"].map(adi).round(3)
    meta.to_csv(RESULTS / "selected_skus.csv", index=False)
    (RESULTS / "inventory_assumptions.json").write_text(
        json.dumps(inv_params, indent=2), encoding="utf-8")

    _example_figure(std, inv, chosen[0])
    return std, inv, meta


def _example_figure(std: pd.DataFrame, inv: pd.DataFrame, sku: str):
    """Sanity-check plot: weekly sales vs simulated start-of-week inventory for one SKU."""
    d = std[std["series_id"] == sku]
    di = inv[inv["series_id"] == sku]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(d["date"], d["sales"], width=5, color="#3a7bd5", label="weekly sales (units)")
    ax.plot(di["date"], di["inventory"], color="#d5573a", lw=1.6,
            label="simulated inventory (start of week)")
    ax.set_title(f"Adapter output example - SKU {sku}: sales vs simulated inventory")
    ax.set_xlabel("Week")
    ax.set_ylabel("Units")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_adapter_example.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    std, inv, meta = build()
    print(f"[adapter] standard format: {std.shape[0]:,} rows = "
          f"{std['series_id'].nunique()} series x {std['date'].nunique()} weeks")
    print(f"[adapter] week range: {std['date'].min().date()} -> {std['date'].max().date()} "
          f"(partial edge weeks trimmed)")
    print(f"[adapter] wrote standard_weekly.parquet/.csv (series_id, date, sales, price), "
          f"inventory_sim.parquet, results/selected_skus.csv, inventory_assumptions.json, "
          f"fig_adapter_example.png")
    print("\n[adapter] selected series (TRAIN-period revenue + ADI):")
    print(meta.to_string(index=False))
    print("\n[adapter] standard-format sample:")
    print(std.head(6).to_string(index=False))
