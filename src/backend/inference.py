"""
inference.py - load the Colab-trained artifacts once and serve MC-Dropout forecasts.

This is the ONLY place the model runs online, and it is inference-only (hard rule: no
training at demo time). It rebuilds the exact training-time inputs from an uploaded
history slice: per-series z-scoring with the TRAIN-period scaler stats, the shared
calendar features (features.py - identical code to training), and the SKU embedding
index from the series map. Unknown SKUs are rejected by design (no cold-start: the
embedding and the per-series scaler both require a known StockCode - see CLAUDE.md).

Uncertainty mirrors the offline evaluation exactly:
    per-step   sigma = sqrt( MC std^2            + residual std^2 )         (chart band)
    lead-time  sigma = sqrt( MC path-sum std^2   + residual sum std^2 )     (rule engine)
"""
import json
import os

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import keras

from ..config import PROCESSED, HORIZON
from ..predictive.features import LOOKBACK, MC_SAMPLES, calendar_features


class Forecaster:
    def __init__(self):
        self.model = keras.saving.load_model(PROCESSED / "lstm_global.keras")
        art = np.load(PROCESSED / "lstm_artifacts.npz")
        self.mu = art["mu"]                       # (n_series, 1) train-period means
        self.sd = art["sd"]                       # (n_series, 1) train-period stds
        self.sig_step = art["sig_resid_step"]     # (H,) aleatoric per step (std space)
        self.sig_cum = art["sig_resid_cum"]       # (H,) aleatoric per lead-time sum
        self.series_map = json.loads(
            (PROCESSED / "lstm_series_map.json").read_text(encoding="utf-8"))

    def known(self, series_id: str) -> bool:
        return series_id in self.series_map

    def forecast(self, series_id: str, dates, sales, n: int = MC_SAMPLES):
        """History (>= LOOKBACK weekly points) -> (forecast_dates, mu, sigma, sigma_cum).

        mu[h]        mean demand h+1 weeks ahead (units),
        sigma[h]     predictive std of that single week,
        sigma_cum[h] predictive std of the SUM of weeks 1..h+1 (MC path sums - the
                     lead-time uncertainty the prescriptions consume).
        """
        j = self.series_map[series_id]
        dates = pd.DatetimeIndex(dates)[-LOOKBACK:]
        z = (np.asarray(sales, "float32")[-LOOKBACK:] - self.mu[j, 0]) / self.sd[j, 0]
        xs = np.concatenate([z[:, None], calendar_features(dates)], axis=1)[None, ...]
        xk = np.array([[j]], "int32")

        preds = np.stack([
            keras.ops.convert_to_numpy(self.model([xs, xk], training=True))[0]
            for _ in range(n)])                                    # (n, H), std space
        mean_z = preds.mean(0)
        tot_z = np.sqrt(preds.std(0) ** 2 + self.sig_step ** 2)
        tot_cum_z = np.sqrt(preds.cumsum(1).std(0) ** 2 + self.sig_cum ** 2)

        fc_dates = [dates[-1] + pd.Timedelta(weeks=h) for h in range(1, HORIZON + 1)]
        return (fc_dates,
                mean_z * self.sd[j, 0] + self.mu[j, 0],
                tot_z * self.sd[j, 0],
                tot_cum_z * self.sd[j, 0])
