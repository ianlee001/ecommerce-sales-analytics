"""
features.py - the input-feature definition SHARED by offline training and the online API.

lstm_global.py (offline, Colab) and src/backend/inference.py (online, Flask) both import
LOOKBACK, MC_SAMPLES and calendar_features from here, so the window length and the
calendar features fed to the model at serving time are BY CONSTRUCTION identical to the
ones it was trained on - they cannot drift apart. This module must stay import-light
(numpy + pandas only, no TensorFlow): the adapter and API import it too.
"""
import numpy as np
import pandas as pd

LOOKBACK = 12       # W: weeks of history fed into the model (a quarter of context)
MC_SAMPLES = 50     # N Monte-Carlo Dropout forward passes


def calendar_features(weeks):
    """Seasonal-phase features per week, identical across series; shape (n_weeks, 3):
    week-of-year as sin/cos (smooth yearly phase) + a weeks-to-Christmas countdown in
    [0, 1) (0 = Christmas week). The countdown is sharper than the smooth sine around
    the one event that dominates this retailer's year (see the EDA figure)."""
    idx = pd.DatetimeIndex(weeks)
    doy = idx.dayofyear.values.astype("float32")
    angle = 2 * np.pi * doy / 365.25
    xmas = pd.to_datetime(pd.DataFrame({"year": idx.year, "month": 12, "day": 25}))
    days = (xmas.values - idx.values).astype("timedelta64[D]").astype(float)
    days = np.where(days < 0, days + 365.25, days)         # already past Xmas -> next year's
    wtc = (days / 7.0 / 52.18).astype("float32")
    return np.stack([np.sin(angle), np.cos(angle), wtc], axis=1).astype("float32")
