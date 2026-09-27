"""
config.py - single source of truth shared by EVERY component.

The adapter, the baselines, the LSTM and the prescriptive layer all import these
constants, so the whole pipeline is guaranteed to agree on ONE time-ordered split, one
horizon and one lead time (hard rules - see CLAUDE.md). Keeping the split constants here
is also what lets the adapter select the Top-N series on the TRAINING period only,
without peeking at the test period.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "online_retail_II.xlsx"
PROCESSED = ROOT / "data" / "processed"
STANDARD = PROCESSED / "standard_weekly.parquet"     # adapter output: series_id, date, sales, price
INVENTORY_SIM = PROCESSED / "inventory_sim.parquet"  # simulated inventory (an ASSUMPTION, not data)
RESULTS = ROOT / "results"
USERS_DB = ROOT / "data" / "users.db"                # auth store (SQLite, gitignored)

FREQ = "W-MON"     # weekly buckets anchored on Monday (matches the adapter)
HORIZON = 4        # forecast horizon: how many weeks ahead we predict
LEAD_TIME = 2      # replenishment lead time (weeks), <= HORIZON: reorder decisions cover this window
TEST_WEEKS = 26    # most-recent weeks held out for evaluation (~6 months), never seen in training
CV_STEP = 1        # rolling-origin step: re-forecast every week across the test period

# Number of rolling-origin windows so the forecasts cover exactly the held-out TEST_WEEKS.
N_WINDOWS = TEST_WEEKS - HORIZON + 1   # = 23

TOP_N = 20         # number of series to model (CLAUDE.md: start ~15-30)
ADI_MAX = 1.32     # weekly intermittency cutoff (Syntetos-Boylan): keep only series dense
                   # enough to forecast at weekly granularity (documented eligibility rule)
