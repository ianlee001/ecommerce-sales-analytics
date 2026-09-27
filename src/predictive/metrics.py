"""
metrics.py - forecast error metrics, reused by the ARIMA baseline and the LSTM.

All take 1-D arrays of actuals `y` and predictions `yhat` and return a plain float so the
results are easy to round and tabulate (rounding for display is a hard rule).

MAPE is undefined when an actual is 0, so we compute it over non-zero actuals only and
also report WAPE (weighted absolute % error), which has no zero-division problem and is
the more honest aggregate for intermittent retail demand.
"""
import numpy as np


def _arrays(y, yhat):
    return np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)


def rmse(y, yhat) -> float:
    y, yhat = _arrays(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat) -> float:
    y, yhat = _arrays(y, yhat)
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat) -> float:
    """Mean absolute percentage error over NON-ZERO actuals (percent)."""
    y, yhat = _arrays(y, yhat)
    nz = y != 0
    return float(np.mean(np.abs((y[nz] - yhat[nz]) / y[nz])) * 100) if nz.any() else float("nan")


def wape(y, yhat) -> float:
    """Weighted absolute percentage error = sum|err| / sum|actual| (percent)."""
    y, yhat = _arrays(y, yhat)
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - yhat)) / denom * 100) if denom else float("nan")


def all_metrics(y, yhat) -> dict:
    return {"rmse": rmse(y, yhat), "mae": mae(y, yhat),
            "mape": mape(y, yhat), "wape": wape(y, yhat)}
