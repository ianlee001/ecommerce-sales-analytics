"""Export the project's cleaned whole-shop transactions for Power BI Desktop.

Run: python powerbi/export_data.py
Outputs live in data/powerbi/, which is already ignored by Git.
The original data, model inputs, and cleaning rules are preserved.
"""
import json
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def cleaned_transactions():
    """Apply the adapter's existing rules using pandas only (no plotting imports).

    The cache contains the original combined sheets, BEFORE deduplication.
    Do not narrow columns before deduplication: that would change row identity.
    """
    cache = ROOT / "data/processed/eda_cache_combined.parquet"
    raw = None
    if cache.exists():
        try:
            raw = pd.read_parquet(cache)
        except ImportError:
            # Parquet is optional; an environment with pandas + openpyxl can
            # read the original workbook without installing the model stack.
            pass
    if raw is None:
        sheets = pd.read_excel(ROOT / "data/raw/online_retail_II.xlsx", sheet_name=None)
        raw = pd.concat(sheets.values(), ignore_index=True)
        for column in raw.select_dtypes(include="object").columns:
            raw[column] = raw[column].astype(str)
    raw["Invoice"] = raw["Invoice"].astype(str)
    raw["StockCode"] = raw["StockCode"].astype(str)
    raw = raw.drop_duplicates()
    raw = raw[~raw["Invoice"].str.startswith("C")]
    raw = raw[(raw["Quantity"] > 0) & (raw["Price"] > 0)]
    raw = raw[raw["StockCode"].str.match(r"^\d{5}[A-Za-z]{0,2}$")].copy()
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
    raw["Revenue"] = raw["Quantity"] * raw["Price"]
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-existing", action="store_true",
                        help="Normalize the existing verified export without rereading Excel")
    args = parser.parse_args()
    output = ROOT / "data" / "powerbi"
    output.mkdir(parents=True, exist_ok=True)
    if args.from_existing:
        transactions = pd.read_csv(output / "sales_cleaned.csv",
                                   dtype={"InvoiceNo": str, "StockCode": str}).rename(
            columns={"InvoiceNo": "Invoice", "UnitPrice": "Price", "ProductName": "Description"})
        transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"])
        prior = json.loads((output / "expected_totals.json").read_text(encoding="utf-8"))
        source_spellings = prior.get("source_stock_code_spellings", prior["products"])
    else:
        transactions = cleaned_transactions()
        source_spellings = int(transactions["StockCode"].nunique())

    # Power BI's data model compares text keys without regard to letter case.
    # Normalize only the BI export AFTER full-field transaction deduplication;
    # preserve every sale and leave raw data / LSTM identifiers unchanged.
    transactions["StockCode"] = transactions["StockCode"].str.strip().str.upper()

    # One descriptive label per stock code. Variants in the raw descriptions are
    # resolved by frequency; no product categories are inferred or invented.
    descriptions = transactions[["StockCode", "Description"]].copy()
    descriptions["Description"] = descriptions["Description"].fillna("").astype(str).str.strip()
    valid = ~descriptions["Description"].str.lower().isin(["", "nan", "none"])
    label_counts = (descriptions[valid]
                    .groupby(["StockCode", "Description"]).size()
                    .rename("Count").reset_index()
                    .sort_values(["StockCode", "Count", "Description"],
                                 ascending=[True, False, True]))
    labels = label_counts.drop_duplicates("StockCode").set_index("StockCode")["Description"]
    products = pd.DataFrame({"StockCode": sorted(transactions["StockCode"].unique())})
    products["ProductName"] = products["StockCode"].map(labels).fillna(products["StockCode"])
    names = products.set_index("StockCode")["ProductName"]

    # A transaction line is not an order. Keep InvoiceNo for DISTINCTCOUNT in DAX.
    sales = transactions[["Invoice", "StockCode", "InvoiceDate", "Quantity",
                          "Price", "Country", "Revenue"]].rename(
        columns={"Invoice": "InvoiceNo", "Price": "UnitPrice"}).copy()
    sales.insert(2, "ProductName", sales["StockCode"].map(names))
    sales.insert(4, "Date", sales["InvoiceDate"].dt.strftime("%Y-%m-%d"))
    sales["InvoiceDate"] = sales["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")
    sales["Revenue"] = sales["Revenue"].round(4)

    # Complete calendar years allow a contiguous, unique Date dimension.
    first_year = transactions["InvoiceDate"].dt.year.min()
    last_year = transactions["InvoiceDate"].dt.year.max()
    dates = pd.date_range(f"{first_year}-01-01", f"{last_year}-12-31")
    calendar = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Year": dates.year,
        "Quarter": "Q" + dates.quarter.astype(str),
        "MonthNumber": dates.month,
        "MonthName": dates.strftime("%b"),
        "YearMonth": dates.strftime("%Y-%m"),
        "YearMonthSort": dates.year * 100 + dates.month,
        "WeekStart": (dates - pd.to_timedelta(dates.dayofweek, unit="D")).strftime("%Y-%m-%d"),
    })
    for name, table in (("sales_cleaned.csv", sales),
                        ("products.csv", products), ("calendar.csv", calendar)):
        table.to_csv(output / name, index=False, encoding="utf-8-sig", float_format="%.4f")

    expected = {
        "transaction_lines": len(sales),
        "orders": int(sales["InvoiceNo"].nunique()),
        "products": len(products),
        "source_stock_code_spellings": source_spellings,
        "units": int(sales["Quantity"].sum()),
        "revenue_gbp": round(float(sales["Revenue"].sum()), 2),
        "start_date": min(sales["Date"]),
        "end_date": max(sales["Date"]),
    }

    # Reconcile against the independently saved dashboard output.
    saved = json.loads((ROOT / "data/processed/eda_aggregates.json").read_text(encoding="utf-8"))["totals"]
    assert expected["orders"] == saved["invoices"]
    assert expected["source_stock_code_spellings"] == saved["products"]
    assert expected["units"] == saved["units"]
    assert abs(expected["revenue_gbp"] - saved["revenue"]) < 0.01
    assert products["StockCode"].is_unique and calendar["Date"].is_unique
    assert products["StockCode"].str.casefold().is_unique
    assert products["StockCode"].eq(products["StockCode"].str.upper()).all()
    assert set(sales["Date"]).issubset(set(calendar["Date"]))

    # Validate the serialized CSV, including types, row count and revenue precision.
    rows = units = 0
    revenue = 0.0
    invoices = set()
    for chunk in pd.read_csv(output / "sales_cleaned.csv", chunksize=100_000,
                             dtype={"InvoiceNo": str, "StockCode": str}):
        rows += len(chunk)
        units += int(chunk["Quantity"].sum())
        revenue += float(chunk["Revenue"].sum())
        invoices.update(chunk["InvoiceNo"])
        assert chunk["StockCode"].isin(products["StockCode"]).all()
        assert chunk["StockCode"].eq(chunk["StockCode"].str.upper()).all()
    assert rows == expected["transaction_lines"]
    assert units == expected["units"] and len(invoices) == expected["orders"]
    assert abs(round(revenue, 2) - expected["revenue_gbp"]) < 0.01

    (output / "expected_totals.json").write_text(
        json.dumps(expected, indent=2), encoding="utf-8")
    print(json.dumps(expected, indent=2))
    print(f"Exported and verified: {output}")


if __name__ == "__main__":
    main()
