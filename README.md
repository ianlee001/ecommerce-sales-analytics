# E-commerce Sales Analytics & LSTM Forecasting

A sales analytics project using **Python and Power BI** to process over one million retail transactions, build an interactive business intelligence report, and forecast product demand with LSTM.

**Focus:** Power BI · data processing and analysis · data modeling · DAX · interactive visualization · LSTM forecasting.

[Download the Power BI report](powerbi/Ecommerce%20Sales%20Analytics.pbix) · [Power BI documentation](powerbi/README.md) · [DAX measures](powerbi/measures.dax)

## What this project demonstrates

| Area | Implementation |
| --- | --- |
| Data cleaning | Combine two Excel sheets, remove duplicates and cancellations, validate quantities and prices, and filter non-product codes with regular expressions. |
| Power BI | Build a sales overview report with KPI cards, monthly sales trends, revenue by country, Top 10 products, product-name tooltips, and a date-range slicer. |
| Power Query | Import CSV data, promote headers, and configure identifier, date, integer, and currency types. |
| Data modeling | Connect the Sales fact table to Products and calendar dimensions using active one-to-many relationships and single-direction filtering. |
| DAX | Calculate revenue, units sold, distinct orders, average order value, product count, and transaction row count. |
| Data preparation | Aggregate transactions by product and week, fill missing sales periods, rank products by training-period revenue, and filter sparse series. |
| Exploratory analysis | Analyze weekly sales, Christmas seasonality, revenue by market, monthly performance, product rankings, and price versus volume. |
| Visualization | Interactive charts, a sortable and searchable product table, CSV export, and actual-versus-predicted sales charts. |
| Forecasting | Train one stacked LSTM across 20 products using a 12-week history window, calendar features, and product embeddings to predict the next four weeks. |
| Evaluation | Compare LSTM with AutoARIMA, historical average, last-value, and seasonal-naive forecasts on the same rolling test windows. |

## Power BI sales report

Open [`powerbi/Ecommerce Sales Analytics.pbix`](powerbi/Ecommerce%20Sales%20Analytics.pbix) in Power BI Desktop. The saved report includes its imported sales data and semantic model.

The report analyzes **1,003,213 cleaned transaction lines** and includes:

- Revenue, order count, average order value, and units sold cards.
- Monthly revenue trends using the calendar dimension.
- Country revenue analysis and Top 10 product rankings.
- Product names in chart tooltips.
- Date-range filtering and interactions between visuals.
- A transaction-count card for data reconciliation.

![Power BI sales data model](docs/images/powerbi-model.png)

| Full-period metric | Value |
| --- | ---: |
| Revenue | £19,642,676.02 |
| Units sold | 11,187,773 |
| Orders | 39,516 |
| Average order value | £497.08 |
| Products | 4,707 |

Product keys are standardized to uppercase for Power BI's case-insensitive data model. The source contains 4,877 distinct case-sensitive code spellings; normalization resolves 170 repeated keys while preserving every sales transaction.

See [`powerbi/README.md`](powerbi/README.md) for the model, measures, and data refresh instructions. The extracted [`report definitions`](powerbi/report-definition/) make visual configurations, field bindings, sorting, and filters inspectable on GitHub.

## Data pipeline

```text
Online Retail II (Excel)
    -> deduplication and transaction validation
    -> product filtering and validated sales records
    -> Power BI fact and dimension tables -> DAX metrics -> interactive sales report
    -> weekly aggregation and exploratory analysis
    -> time-series windows and LSTM forecasts
    -> interactive charts and inventory what-if scenarios
```

The original run processes **1,067,371 transaction rows**, removes **34,335 exact duplicates**, and retains **1,003,213 rows** after cleaning. Forecasting uses **20 products across 104 complete weeks**. Product selection and scaling use the training period; the most recent **26 weeks** are held out for evaluation.

## Technology

- **Analysis and preparation:** Python, pandas, NumPy, regular expressions, OpenPyXL, Parquet.
- **Business intelligence:** Power BI Desktop, Power Query, DAX, fact and dimension modeling, interactive report filtering.
- **Visualization:** Power BI charts and KPI cards, Matplotlib for analysis figures, Recharts for web charts.
- **Forecasting:** TensorFlow/Keras LSTM; StatsForecast for AutoARIMA and baseline comparisons.
- **Demo application:** React, Flask REST API, and SQLite for local demo accounts.

## Dashboard examples

![Sales overview dashboard](docs/images/dashboard.png)

### Revenue and market analysis

![Revenue by market and monthly performance](docs/images/analysis.png)

### Forecasts against held-out actual sales

![Four-week forecast with observed sales](docs/images/forecast.png)

The web dashboard also includes price-volume exploration and inventory scenarios using simulated stock and elasticity-based demand adjustments.

## Run the local demo

Use Python 3.11–3.13 and Node.js 18 or later with npm. Run commands from the repository root.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
npm --prefix frontend ci
python scripts/prepare_demo.py
```

The preparation script copies the bundled aggregate data and trained model into the ignored `data/processed/` directory. No raw transaction download or model training is needed for the demo.

Start the API and dashboard in separate terminals:

```bash
python -m src.backend.app
```

```bash
npm --prefix frontend run dev
```

Open **http://localhost:5173**. The local demo account is **admin / admin123**. The app creates its local account database on first use. Deployment configuration uses `FYP_SECRET_KEY` and `FYP_ADMIN_CODE`; configure these and replace the demo credentials for a hosted instance.

On the prediction page, choose a bundled example or upload a CSV from `data/processed/examples/`:

- **Validation:** held-out historical weeks with actual sales displayed.
- **Future:** the weeks after the recorded dataset, with no observed outcomes.

Forecast inputs use consecutive weekly history for one of the 20 supported products. Runtime forecasting loads the saved model and performs inference.

## Explore the code

| Path | Purpose |
| --- | --- |
| [`powerbi/`](powerbi/) | Saved Power BI report, DAX measures, report definitions, and data preparation script. |
| [`src/data/adapter.py`](src/data/adapter.py) | Cleaning, validation, product selection, weekly aggregation, and gap filling. |
| [`src/data/eda_aggregates.py`](src/data/eda_aggregates.py) | Revenue, market, month, and product summaries for the dashboard. |
| [`src/predictive/features.py`](src/predictive/features.py) | Shared calendar features and model history length. |
| [`src/predictive/lstm_global.py`](src/predictive/lstm_global.py) | Window preparation, stacked LSTM training, uncertainty, and evaluation. |
| [`src/predictive/baselines.py`](src/predictive/baselines.py) | Baseline forecasting on rolling test windows. |
| [`frontend/src/pages/Dashboard.jsx`](frontend/src/pages/Dashboard.jsx) | Interactive exploratory analysis. |
| [`frontend/src/pages/Predict.jsx`](frontend/src/pages/Predict.jsx) | Forecast visualization and scenarios. |
| `src/prescriptive/` | Supporting inventory rules and scenario calculations. |
| `demo/processed/` | Small aggregate demo datasets, example inputs, and saved model artifacts. |
| `results/` | Selected analysis figures and recorded model comparison. |

## Reproduce the offline pipeline

Download the Excel workbook from [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), and place it at `data/raw/online_retail_II.xlsx`. Then run:

```bash
python -m src.data.adapter
python -m src.predictive.baselines
python -m src.predictive.lstm_global
python -m src.prescriptive.elasticity
python -m src.data.eda_aggregates
python -m src.data.make_examples
```

The saved comparison in [`results/comparison_models.csv`](results/comparison_models.csv) records LSTM RMSE **363.38** and AutoARIMA RMSE **366.67**; their MAE values are **202.23** and **186.95**, respectively.

The LSTM uses Monte Carlo Dropout and a validation-residual adjustment to produce forecast bands.

## Data attribution

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D).

The source dataset is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The Power BI report contains cleaned transaction data derived from this public dataset. The Python demo bundle contains aggregated product/week data and model artifacts. See [`powerbi/README.md`](powerbi/README.md) and [`demo/README.md`](demo/README.md) for the transformations.
