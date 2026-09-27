# Bundled demo artifacts

These small artifacts let the dashboard run without downloading the raw Excel workbook or training the model. Run `python scripts/prepare_demo.py` from the repository root to copy them into the ignored local data directory. Existing files are preserved.

## Data source and attribution

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D

Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii

Dataset license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

Changes from the original: combine sheets, remove duplicates/cancellations/invalid transactions/non-product codes, select products using training-period revenue and density, aggregate sales and prices weekly, trim partial edge weeks, and fill gaps. The bundle contains no customer identifiers or invoice-level records.

| Artifact | Description |
| --- | --- |
| `standard_weekly.csv` / `.parquet` | 20 selected products over 104 weeks; product ID, date, units, and mean price. |
| `eda_aggregates.json` | Whole-shop totals, market/month summaries, weekly totals, and product summaries. |
| `examples/` | Twelve-week input histories for validation and future forecasting demos. |
| `inventory_sim.parquet` | Simulated inventory, not observed stock. |
| `elasticity.json` | Estimated price elasticities and fallback assumptions for scenarios. |
| `lstm_global.keras` | Saved global LSTM trained for this project. |
| `lstm_artifacts.npz` | Per-product scaling parameters and residual uncertainty estimates. |
| `lstm_series_map.json` | Mapping from the supported product codes to embedding indices. |

The model uses sales, calendar features, and product identity. Price is only used in a separate scenario calculation. Forecasts beyond the recorded history have no known actual outcomes in this dataset.
