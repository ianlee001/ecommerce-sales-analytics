"""
lstm_global.py - the global stacked LSTM with Monte-Carlo-Dropout uncertainty (step 4).

Design (locked in CLAUDE.md):
  * ONE global model across all Top-N series (not one per SKU) - it borrows strength
    across series and stays lightweight.
  * Input = the last W weeks of a series (per-series standardised) + seasonal-phase
    calendar features (week-of-year sin/cos + a weeks-to-Christmas countdown: the EDA
    shows one dominant Q4 peak that a 12-week window cannot see coming on its own)
    + the SKU identity (an embedding). Output = the next H weeks (direct multi-step).
    Price is NOT an input (its effect belongs to the elasticity what-if layer).
  * Uncertainty = Monte-Carlo Dropout (dropout ACTIVE at inference, N passes) PLUS an
    aleatoric residual term measured on the validation slice, combined in quadrature
    (sigma^2 = sigma_MC^2 + sigma_resid^2). MC-Dropout alone captures only model
    (epistemic) uncertainty and is far too tight on noisy weekly demand.
  * CALIBRATION CHECK (printed): the P10-P90 band must cover ~80% of test actuals and
    the 95% band ~95% - the rule engine must not consume an uncalibrated sigma.
  * Lead-time totals: the rule engine needs demand SUMMED over the lead time, so we also
    output the std of the MC SAMPLE-PATH SUMS (cumulative over the horizon). This
    preserves week-to-week error correlation - no independence assumption.

Fairness / hard rules:
  * Identical time-ordered split and rolling origins to the baselines (src/config.py).
  * Per-series scaling stats from the TRAIN period only. Hyperparameters and early
    stopping use a time-ordered validation slice INSIDE train; the test slice is touched
    once, for the final evaluation.
  * The LSTM trains once; at each origin it is fed the observed window up to that week
    (using past actuals as inputs is not leakage). ARIMA refits at every origin - a
    deliberate, documented asymmetry that if anything favours ARIMA.
  * DECISIVE CHECK (printed): the LSTM must beat the per-series mean (HistoricAverage).
    If not, there is no usable signal - stop and investigate before building on top.

Artifacts saved for the Flask API (serve by inference, never retrain at demo time):
    lstm_global.keras, lstm_artifacts.npz (per-series mu/sd + residual stds),
    lstm_series_map.json (StockCode -> embedding index).

Run from the repo root:  python -m src.predictive.lstm_global
"""
import json
import os
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# quiet + deterministic
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..config import STANDARD, RESULTS, HORIZON, TEST_WEEKS, N_WINDOWS
from . import metrics as M
from .features import LOOKBACK, MC_SAMPLES, calendar_features

# --------------------------------------------------------------------------------------
# Hyper-parameters (tunable ON THE VALIDATION SLICE ONLY; record every configuration tried)
# --------------------------------------------------------------------------------------
# LOOKBACK (W), MC_SAMPLES and calendar_features live in features.py - shared with the
# Flask API so offline training and online inference can never drift apart.
VAL_WEEKS = 12      # time-ordered validation slice at the end of the train period
EMB_DIM = 4         # SKU embedding size
LSTM_UNITS = (64, 32)
DROPOUT = 0.2       # also the MC-Dropout rate at inference
EPOCHS = 100
BATCH = 32
SEED = 0
BASELINES = ["AutoARIMA", "HistoricAverage", "Naive", "SeasonalNaive"]


def _seed():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


# --------------------------------------------------------------------------------------
# Data prep
# --------------------------------------------------------------------------------------
def load_matrix():
    """Return (S, series_ids, weeks): S is a (n_series x n_weeks) units matrix."""
    df = pd.read_parquet(STANDARD)
    weeks = np.sort(df["date"].unique())
    series_ids = sorted(df["series_id"].unique())
    piv = df.pivot(index="series_id", columns="date", values="sales").loc[series_ids, weeks]
    return piv.values.astype("float32"), series_ids, pd.to_datetime(weeks)


def standardise(S, test_start):
    """Per-series z-score using TRAIN-period stats only. Returns Z, mu, sd."""
    mu = S[:, :test_start].mean(axis=1, keepdims=True)
    sd = S[:, :test_start].std(axis=1, keepdims=True) + 1e-6
    return (S - mu) / sd, mu, sd


def _window(Z, C, j, a, b):
    """One input window for series j over weeks [a:b): [std sales, calendar features]."""
    return np.concatenate([Z[j, a:b, None], C[a:b]], axis=1)


def make_training_windows(Z, C, test_start):
    """All (input window, sku, target) samples whose TARGET lies inside the train period."""
    Xs, Xk, Y = [], [], []
    n_series = Z.shape[0]
    val_start = test_start - VAL_WEEKS
    tr_idx, va_idx = [], []
    for j in range(n_series):
        # last input week index = i+W-1; target weeks = i+W .. i+W+H-1, must be < test_start
        for i in range(0, test_start - LOOKBACK - HORIZON + 1):
            Xs.append(_window(Z, C, j, i, i + LOOKBACK))
            Xk.append(j)
            Y.append(Z[j, i + LOOKBACK:i + LOOKBACK + HORIZON])
            # route to val if the target's last week falls in the last VAL_WEEKS of train
            (va_idx if (i + LOOKBACK + HORIZON - 1) >= val_start else tr_idx).append(len(Xs) - 1)
    Xs = np.asarray(Xs, "float32")              # (n, W, 1 + n_calendar)
    Xk = np.asarray(Xk, "int32")
    Y = np.asarray(Y, "float32")                # (n, H)
    return (Xs, Xk, Y), np.asarray(tr_idx), np.asarray(va_idx)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def build_model(n_series, n_features):
    seq_in = keras.Input(shape=(LOOKBACK, n_features), name="sequence")
    sku_in = keras.Input(shape=(1,), dtype="int32", name="sku")

    x = layers.LSTM(LSTM_UNITS[0], return_sequences=True)(seq_in)
    x = layers.Dropout(DROPOUT)(x)              # MC-Dropout (kept on at inference)
    x = layers.LSTM(LSTM_UNITS[1])(x)
    x = layers.Dropout(DROPOUT)(x)

    e = layers.Embedding(n_series, EMB_DIM)(sku_in)
    e = layers.Flatten()(e)

    h = layers.Concatenate()([x, e])
    h = layers.Dense(32, activation="relu")(h)
    h = layers.Dropout(DROPOUT)(h)
    out = layers.Dense(HORIZON)(h)              # direct multi-step output

    model = keras.Model([seq_in, sku_in], out)
    model.compile(optimizer="adam", loss="mse")
    return model


def mc_predict(model, Xs, Xk, n=MC_SAMPLES):
    """N stochastic forward passes (dropout ON), in standardised space.
    Returns (mean, per-step std, lead-time-sum std). The sum-std is the std of each MC
    sample path's CUMULATIVE sum, so week-to-week error correlation is preserved - no
    independence assumption (brief s7)."""
    preds = np.stack([model([Xs, Xk], training=True).numpy() for _ in range(n)])  # (n,B,H)
    return preds.mean(0), preds.std(0), preds.cumsum(axis=2).std(0)


def aleatoric_std(model, Xs_va, Xk_va, Y_va):
    """Residual (aleatoric) std on the validation slice, in standardised space.

    MC-Dropout captures only epistemic (model) uncertainty and is far too tight on this
    noisy weekly demand, so on its own the band would be uncalibrated. We measure the
    model's own residuals on the held-out validation slice (inside the train period ->
    no leakage) and combine them with the MC std in quadrature at inference.
    Returned per horizon step AND per cumulative lead-time sum (the rule engine
    consumes lead-time totals)."""
    mean_z, _, _ = mc_predict(model, Xs_va, Xk_va)
    resid = Y_va - mean_z                                   # (n_val, H)
    return (resid.std(axis=0).astype("float32"),
            resid.cumsum(axis=1).std(axis=0).astype("float32"))


# --------------------------------------------------------------------------------------
# Rolling-origin evaluation (same origins as the baselines)
# --------------------------------------------------------------------------------------
def evaluate(model, Z, C, S, mu, sd, series_ids, weeks, test_start, sig_step, sig_cum):
    """At each origin cutoff c, feed weeks [c-W+1..c], forecast c+1..c+H; back-transform.
    Both stds are epistemic (MC) + aleatoric (residual) in quadrature:
      y_std      - std of ONE week's demand (drawn as the chart band),
      y_std_cum  - std of the SUM of weeks 1..h (consumed by the rule engine)."""
    cutoffs = range(test_start - 1, test_start - 1 + N_WINDOWS)   # same origins as ARIMA
    Xs, Xk, keys = [], [], []
    for c in cutoffs:
        for j in range(len(series_ids)):
            Xs.append(_window(Z, C, j, c - LOOKBACK + 1, c + 1))
            Xk.append(j)
            keys.append((j, c))
    Xs = np.asarray(Xs, "float32")
    Xk = np.asarray(Xk, "int32")
    mean_z, std_z, std_cum_z = mc_predict(model, Xs, Xk)
    tot_z = np.sqrt(std_z ** 2 + sig_step[None, :] ** 2)
    tot_cum_z = np.sqrt(std_cum_z ** 2 + sig_cum[None, :] ** 2)

    rows = []
    for k, (j, c) in enumerate(keys):
        for hh in range(HORIZON):
            rows.append(dict(
                series_id=series_ids[j], cutoff=weeks[c], ds=weeks[c + 1 + hh], h=hh + 1,
                y_true=float(S[j, c + 1 + hh]),
                y_pred=float(mean_z[k, hh] * sd[j, 0] + mu[j, 0]),
                y_std=float(tot_z[k, hh] * sd[j, 0]),           # std scales with sd (mu cancels)
                y_std_cum=float(tot_cum_z[k, hh] * sd[j, 0])))  # std of the h-week SUM
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def metric_tables(pred):
    per = (pred.groupby("series_id")
               .apply(lambda g: pd.Series(M.all_metrics(g["y_true"], g["y_pred"])),
                      include_groups=False)
               .reset_index())
    macro = per[["rmse", "mae", "mape", "wape"]].mean()
    micro = pd.Series(M.all_metrics(pred["y_true"], pred["y_pred"]))
    by_h = (pred.groupby("h")
                .apply(lambda g: pd.Series(M.all_metrics(g["y_true"], g["y_pred"])),
                       include_groups=False)
                .reset_index())
    return per, macro, micro, by_h


def compare_with_baselines(lstm_micro):
    """Pooled LSTM vs ARIMA + the naive trio, on identical folds."""
    b = pd.read_csv(RESULTS / "baseline_cv_predictions.csv")
    rows = {"LSTM": lstm_micro}
    for m in BASELINES:
        rows[m] = pd.Series(M.all_metrics(b["y"], b[m]))
    return pd.DataFrame(rows).T[["rmse", "mae", "mape", "wape"]], b


def uncertainty_figure(pred, sid):
    g = pred[(pred["series_id"] == sid) & (pred["h"] == 1)].sort_values("ds")
    lo = (g["y_pred"] - 1.96 * g["y_std"]).clip(lower=0)   # demand cannot be negative
    hi = g["y_pred"] + 1.96 * g["y_std"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(g["ds"], g["y_true"], color="#3a7bd5", lw=1.4, label="actual")
    ax.plot(g["ds"], g["y_pred"], color="#d5573a", lw=1.6, ls="--", label="LSTM mean (1-wk-ahead)")
    ax.fill_between(g["ds"], lo, hi, color="#d5573a", alpha=0.2,
                    label="95% predictive band (MC + residual)")
    ax.set_title(f"Global LSTM + MC-Dropout - SKU {sid}: forecast & uncertainty (test period)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Units")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_lstm_example.png", dpi=120)
    plt.close(fig)


def main():
    _seed()
    S, series_ids, weeks = load_matrix()
    test_start = len(weeks) - TEST_WEEKS
    Z, mu, sd = standardise(S, test_start)
    C = calendar_features(weeks)
    (Xs, Xk, Y), tr, va = make_training_windows(Z, C, test_start)
    print(f"[lstm] {len(series_ids)} series x {len(weeks)} weeks | train windows {len(tr)}, "
          f"val {len(va)} | W={LOOKBACK} H={HORIZON} feats={Xs.shape[-1]}")

    model = build_model(len(series_ids), Xs.shape[-1])
    es = keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True, monitor="val_loss")
    hist = model.fit([Xs[tr], Xk[tr]], Y[tr],
                     validation_data=([Xs[va], Xk[va]], Y[va]),
                     epochs=EPOCHS, batch_size=BATCH, verbose=0, callbacks=[es])
    print(f"[lstm] trained {len(hist.history['loss'])} epochs | "
          f"best val_loss {min(hist.history['val_loss']):.4f}")

    sig_step, sig_cum = aleatoric_std(model, Xs[va], Xk[va], Y[va])
    print(f"[lstm] residual std per step (std space): {np.round(sig_step, 3)} | "
          f"per lead-time sum: {np.round(sig_cum, 3)}")

    # ---- artifacts for the Flask API (inference-only serving) ----
    model.save(STANDARD.parent / "lstm_global.keras")
    np.savez(STANDARD.parent / "lstm_artifacts.npz", mu=mu, sd=sd,
             sig_resid_step=sig_step, sig_resid_cum=sig_cum)
    (STANDARD.parent / "lstm_series_map.json").write_text(
        json.dumps({s: i for i, s in enumerate(series_ids)}, indent=2), encoding="utf-8")

    pred = evaluate(model, Z, C, S, mu, sd, series_ids, weeks, test_start, sig_step, sig_cum)
    pred.to_csv(RESULTS / "lstm_cv_predictions.csv", index=False)

    # ---- calibration: the rule engine must not consume an uncalibrated sigma ----
    err = np.abs(pred["y_true"] - pred["y_pred"])
    cov80 = float((err <= 1.2816 * pred["y_std"]).mean())
    cov95 = float((err <= 1.96 * pred["y_std"]).mean())
    print(f"[lstm] CALIBRATION - band coverage on test: "
          f"P10-P90 {cov80:.1%} (target ~80%) | 95% {cov95:.1%} (target ~95%)")

    per, macro, micro, by_h = metric_tables(pred)
    per.round(3).to_csv(RESULTS / "lstm_metrics.csv", index=False)
    comp, b = compare_with_baselines(micro)
    comp.round(3).to_csv(RESULTS / "comparison_models.csv")
    uncertainty_figure(pred, per.sort_values("rmse").iloc[len(per) // 2]["series_id"])

    print("\n[lstm] aggregate over the held-out test period:")
    print(f"  macro: RMSE {macro['rmse']:.2f} | MAE {macro['mae']:.2f} "
          f"| MAPE {macro['mape']:.1f}% | WAPE {macro['wape']:.1f}%")
    print(f"  micro: RMSE {micro['rmse']:.2f} | MAE {micro['mae']:.2f} "
          f"| MAPE {micro['mape']:.1f}% | WAPE {micro['wape']:.1f}%")
    print("\n[lstm] pooled comparison on identical folds (lower is better):")
    print(comp.round(2).to_string())

    mean_rmse = float(comp.loc["HistoricAverage", "rmse"])
    lstm_rmse = float(comp.loc["LSTM", "rmse"])
    verdict = "PASS" if lstm_rmse < mean_rmse else "FAIL - investigate before building further"
    print(f"\n[lstm] DECISIVE CHECK (beat the per-series mean): "
          f"LSTM RMSE {lstm_rmse:.1f} vs HistoricAverage {mean_rmse:.1f} -> {verdict}")

    a_by_h = (b.groupby("h")
                .apply(lambda g: pd.Series(M.all_metrics(g["y"], g["AutoARIMA"])),
                       include_groups=False)
                .reset_index())
    merged = by_h.merge(a_by_h, on="h", suffixes=("_LSTM", "_ARIMA"))
    print("\n[lstm] error by forecast step (weeks ahead):")
    print(merged[["h", "rmse_LSTM", "rmse_ARIMA", "wape_LSTM", "wape_ARIMA"]]
          .round(2).to_string(index=False))
    print("\n[lstm] wrote lstm_metrics.csv, comparison_models.csv, lstm_cv_predictions.csv, "
          "fig_lstm_example.png + artifacts (lstm_global.keras, lstm_artifacts.npz, "
          "lstm_series_map.json)")


if __name__ == "__main__":
    main()
