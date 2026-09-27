"""
make_examples.py - bundle example upload slices for the prediction page (demo safety).

The dashboard's ONLY upload is a template-conforming HISTORY slice of one StockCode
(standard-format columns `series_id, date, sales, price`; >= LOOKBACK consecutive weekly
rows; the SKU must be one of the modelled Top-N - no cold-start, see CLAUDE.md).

TWO demo modes per SKU, matching the two stories the demo must tell:
  * validation - history ends at the LAST rolling-origin cutoff, so the forecast horizon
    still has HELD-OUT actuals to overlay (the green line): "the model never saw these
    weeks - here is what it predicted vs what really happened" = the accuracy evidence.
  * future     - history ends at the LAST recorded week, so the forecast extends BEYOND
    the dataset: a genuine unknown future, no actuals exist - what a real SME would see
    in deployment.

Files go to data/processed/examples/ (served by /api/example) AND data/demo_uploads/
(same content, friendlier names - for manually uploading through the UI in a demo).

Run from the repo root:  python -m src.data.make_examples
"""
import pandas as pd

from ..config import ROOT, STANDARD, PROCESSED, HORIZON
from ..predictive.features import LOOKBACK

EXAMPLE_SKUS = ["22411", "84879", "20725"]   # median / large / steady - varied stories
DEMO_DIR = ROOT / "data" / "demo_uploads"


def main():
    df = pd.read_parquet(STANDARD)
    weeks = sorted(df["date"].unique())
    ends = {"validation": weeks[-HORIZON - 1],   # horizon still has held-out actuals
            "future": weeks[-1]}                 # forecast goes beyond recorded history
    outdir = PROCESSED / "examples"
    outdir.mkdir(exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    for sku in EXAMPLE_SKUS:
        for mode, end in ends.items():
            g = (df[(df["series_id"] == sku) & (df["date"] <= end)]
                 .sort_values("date").tail(LOOKBACK))
            g.to_csv(outdir / f"{mode}_{sku}.csv", index=False)
            g.to_csv(DEMO_DIR / f"demo_SKU{sku}_{mode}.csv", index=False)
            print(f"[examples] {mode}_{sku}: {len(g)} weeks "
                  f"{g['date'].min().date()} -> {g['date'].max().date()}")
    print(f"[examples] wrote {outdir} (API) and {DEMO_DIR} (manual-upload demo files)")


if __name__ == "__main__":
    main()
