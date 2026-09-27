"""
app.py - Flask REST API (build-order step 9): serves the pre-trained model by inference.

No training happens here (hard rule: demo = inference + display on the held-out slice).
Loaded once at startup: the trained LSTM + per-series scalers + series map (the Colab
artifacts), the standard-format weekly data (READ-ONLY), the simulated inventory
trajectory, the elasticity artifact, and the bundled example slices.

Endpoints (JSON; every number rounded for display - hard rule). All data endpoints
require a Bearer token from /api/auth/login (auth.py; /api/health stays open):
    GET  /api/health              liveness + what is loaded (open)
    POST /api/auth/register|login GET /api/me  POST /api/auth/change-password
    GET|DELETE /api/users[/<id>]  thinnest admin shell (admin role)
    GET  /api/series              modelled SKUs (+ which have bundled examples)
    GET  /api/examples            bundled demo slices (validation + future mode per SKU)
    GET  /api/example/<sku>?mode= one bundled example slice ("use example data" button)
    GET  /api/eda/dashboard       everything the 4-tab landing page needs
    GET|POST /api/messages        message board (admin composes; users read theirs)
    POST /api/forecast            upload one SKU's history slice -> MC-Dropout forecast
    POST /api/prescribe           closed-form what-if: discount / lead time / service
                                  level / current inventory -> prescription

Interaction contract (implementation-plan C): /api/forecast runs MC Dropout ONCE and
returns mu / sigma / sigma_cum; the dashboard sliders then call /api/prescribe, which
recomputes prescriptions closed-form from those cached numbers + the elasticity formula
Q' = Q * (P'/P)^epsilon - the model is never re-run by a slider.

Run from the repo root:  python -m src.backend.app     (http://127.0.0.1:5000)
"""
import json
import math
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

from ..config import STANDARD, INVENTORY_SIM, PROCESSED, RESULTS, HORIZON
from ..predictive.features import LOOKBACK
from ..prescriptive.rules import PolicyParams, prescribe
from .auth import init_db, register_auth_routes, require_auth
from .inference import Forecaster
from .messages import init_messages, register_message_routes

app = Flask(__name__)
init_db()
init_messages()
register_auth_routes(app)
register_message_routes(app)

# ---- loaded once at startup (retail data is READ-ONLY - no CRUD anywhere) ----
FC = Forecaster()
STD = pd.read_parquet(STANDARD)
INV = pd.read_parquet(INVENTORY_SIM)
ELAST = json.loads((PROCESSED / "elasticity.json").read_text(encoding="utf-8"))
SKUS = pd.read_csv(RESULTS / "selected_skus.csv", dtype={"series_id": str})
BUSINESS = json.loads((PROCESSED / "eda_aggregates.json").read_text(encoding="utf-8"))
ELAST_TAB = pd.read_csv(RESULTS / "elasticity.csv", dtype={"series_id": str})
EXAMPLES_DIR = PROCESSED / "examples"
EXAMPLES = sorted(
    ({"mode": p.stem.split("_", 1)[0], "sku": p.stem.split("_", 1)[1]}
     for p in EXAMPLES_DIR.glob("*_*.csv")),
    key=lambda e: (e["sku"], e["mode"] != "validation"))
EXAMPLE_SKUS = {e["sku"] for e in EXAMPLES}
Z_BY_SERVICE = {90: 1.2816, 95: 1.6449, 99: 2.3263}


@app.after_request
def cors(resp):
    """Allow the React dev server (different port) to call the API."""
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    return resp


def bad(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


def epsilon_for(series_id: str):
    """Per-series elasticity where the pre-specified criteria passed, else the
    documented literature fallback."""
    if series_id in ELAST["per_series"]:
        return ELAST["per_series"][series_id], "estimated"
    return ELAST["default"], "literature_fallback"


# --------------------------------------------------------------------------------------
# Read endpoints
# --------------------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "series_loaded": len(FC.series_map),
        "weeks": int(STD["date"].nunique()),
        "examples": EXAMPLES,
        "horizon_weeks": HORIZON,
        "min_history_weeks": LOOKBACK,
    })


@app.get("/api/examples")
@require_auth()
def examples_list():
    """Bundled demo slices: per SKU a 'validation' (held-out actuals exist -> accuracy
    evidence) and a 'future' (beyond recorded history -> genuine unknown) mode."""
    return jsonify(EXAMPLES)


@app.get("/api/series")
@require_auth()
def series():
    rows = []
    for r in SKUS.itertuples():
        eps, src = epsilon_for(r.series_id)
        rows.append({"series_id": r.series_id,
                     "train_revenue": round(float(r.train_revenue), 2),
                     "train_adi": round(float(r.train_adi), 3),
                     "epsilon": eps, "epsilon_source": src,
                     "has_example": r.series_id in EXAMPLE_SKUS})
    return jsonify(rows)


@app.get("/api/example/<sku>")
@require_auth()
def example(sku: str):
    mode = request.args.get("mode", "validation")
    path = EXAMPLES_DIR / f"{mode}_{sku}.csv"
    if not path.exists():
        return bad(f"no bundled '{mode}' example for SKU {sku}", 404)
    df = pd.read_csv(path, dtype={"series_id": str})
    return jsonify(df.to_dict("records"))


@app.get("/api/eda/dashboard")
@require_auth()
def eda_dashboard():
    """Everything the 4-tab landing page needs, in one call: business headline numbers
    (offline aggregates from the cleaned raw data), the weekly total series, the product
    table (joined with the latest SIMULATED stock - a documented assumption), and the
    price-insight block (price/volume scatter + elasticity-based discount readiness)."""
    per = (STD.groupby("series_id")
              .agg(mean=("sales", "mean"), avg_price=("price", "mean")).reset_index())
    latest_inv = INV[INV["date"] == INV["date"].max()].set_index("series_id")["inventory"]

    product_table = []
    for row in BUSINESS["product_table"]:
        stock = float(latest_inv.get(row["series_id"], 0.0))
        product_table.append({**row, "sim_stock": round(stock),
                              "est_value": round(stock * row["avg_price"], 2)})

    scatter = [{"series_id": r.series_id,
                "avg_price": round(float(r.avg_price), 2),
                "avg_weekly_sales": round(float(r.mean), 1)} for r in per.itertuples()]
    readiness = []
    for r in ELAST_TAB.itertuples():
        eps, src = epsilon_for(r.series_id)
        if bool(r.reliable) and eps <= -1.5:
            verdict = "Ready - price-elastic"
        elif bool(r.reliable):
            verdict = "Mild response"
        else:
            verdict = "Not identified - literature fallback"
        readiness.append({"series_id": r.series_id, "epsilon": round(float(eps), 2),
                          "source": src, "price_cv": round(float(r.price_cv), 3),
                          "verdict": verdict})

    return jsonify({
        "business": {"totals": BUSINESS["totals"], "markets": BUSINESS["markets"],
                     "best_months": BUSINESS["best_months"]},
        "date_range": [str(STD["date"].min().date()), str(STD["date"].max().date())],
        "n_weeks": int(STD["date"].nunique()),
        # whole-shop weekly totals (offline aggregate) - same scope as the headline cards
        "total_weekly": BUSINESS["total_weekly"],
        "product_table": product_table,
        "price_insight": {"scatter": scatter, "discount_readiness": readiness},
    })


# --------------------------------------------------------------------------------------
# Forecast (the one model call)
# --------------------------------------------------------------------------------------
def parse_upload():
    """Uploaded CSV file (multipart 'file') or JSON {'rows': [...]}, standard format.
    Returns (series_id, history DataFrame) or raises ValueError with a user message."""
    if "file" in request.files:
        df = pd.read_csv(request.files["file"], dtype={"series_id": str})
    elif request.is_json and "rows" in (request.get_json(silent=True) or {}):
        df = pd.DataFrame(request.get_json()["rows"])
        df["series_id"] = df["series_id"].astype(str)
    else:
        raise ValueError("upload a CSV file (field 'file') or JSON {'rows': [...]}")

    missing = {"series_id", "date", "sales"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)} "
                         f"(template: series_id, date, sales[, price])")
    sids = df["series_id"].unique()
    if len(sids) != 1:
        raise ValueError("the slice must contain exactly ONE StockCode")
    sid = str(sids[0])
    if not FC.known(sid):
        raise ValueError(f"unknown StockCode {sid}: only the modelled Top-N series are "
                         f"supported (no cold-start - see the series list)")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    if df["date"].isna().any() or df["sales"].isna().any() or (df["sales"] < 0).any():
        raise ValueError("dates must parse and sales must be non-negative numbers")
    df = df.sort_values("date").reset_index(drop=True)
    gaps = df["date"].diff().dropna().dt.days
    if len(df) < LOOKBACK:
        raise ValueError(f"need at least {LOOKBACK} weekly rows, got {len(df)}")
    if not (gaps == 7).all():
        raise ValueError("rows must be consecutive weekly buckets (7-day spacing)")
    return sid, df


@app.post("/api/forecast")
@require_auth()
def forecast():
    try:
        sid, hist = parse_upload()
    except ValueError as e:
        return bad(str(e))

    fc_dates, mu, sig, sig_cum = FC.forecast(sid, hist["date"], hist["sales"])

    # overlay ground truth server-side where the held-out data has it (demo slices)
    actuals = STD[(STD["series_id"] == sid) & (STD["date"].isin(fc_dates))]
    truth = {d: float(s) for d, s in zip(actuals["date"], actuals["sales"])}

    # default current inventory: the simulated trajectory at the first forecast week
    inv_row = INV[(INV["series_id"] == sid) & (INV["date"] == fc_dates[0])]
    default_inv = round(float(inv_row["inventory"].iloc[0]), 1) if len(inv_row) else None

    eps, eps_src = epsilon_for(sid)
    return jsonify({
        "series_id": sid,
        "history": [{"date": str(d.date()), "sales": round(float(s), 1)}
                    for d, s in zip(hist["date"], hist["sales"])],
        "forecast": [{"date": str(d.date()),
                      "mu": round(float(m), 1),
                      "sigma": round(float(s), 1),
                      "sigma_cum": round(float(sc), 1),
                      "p10": round(max(0.0, float(m - 1.2816 * s)), 1),
                      "p90": round(float(m + 1.2816 * s), 1),
                      "actual": round(truth[d], 1) if d in truth else None}
                     for d, m, s, sc in zip(fc_dates, mu, sig, sig_cum)],
        "default_inventory": default_inv,
        "epsilon": eps, "epsilon_source": eps_src,
        "lead_time_max": HORIZON,
    })


# --------------------------------------------------------------------------------------
# Prescribe (closed-form what-if - the model is NOT re-run)
# --------------------------------------------------------------------------------------
@app.post("/api/prescribe")
@require_auth()
def prescribe_endpoint():
    b = request.get_json(silent=True) or {}
    try:
        mu = np.asarray(b["mu"], dtype=float)
        sigma = np.asarray(b["sigma"], dtype=float)
        sigma_cum = np.asarray(b["sigma_cum"], dtype=float)
        inv = float(b["current_inventory"])
        lead = int(b.get("lead_time", 2))
        service = int(b.get("service_level", 95))
        discount = float(b.get("discount_pct", 0.0))
        sid = str(b.get("series_id", ""))
    except (KeyError, TypeError, ValueError):
        return bad("body needs: mu[], sigma[], sigma_cum[] (from /api/forecast), "
                   "current_inventory; optional: lead_time, service_level (90|95|99), "
                   "discount_pct, series_id")
    if not (len(mu) == len(sigma) == len(sigma_cum)) or len(mu) == 0:
        return bad("mu, sigma and sigma_cum must be equally long and non-empty")
    if service not in Z_BY_SERVICE:
        return bad("service_level must be one of 90, 95, 99")
    if not 0 <= discount <= 90:
        return bad("discount_pct must be within 0..90")
    lead = max(1, min(lead, len(mu), HORIZON))   # lead time capped at the horizon

    # elasticity what-if: P' = P*(1-d), Q' = Q*(P'/P)^eps; the band scales with the mean
    eps = float(b["epsilon"]) if "epsilon" in b else epsilon_for(sid)[0]
    factor = (1.0 - discount / 100.0) ** eps if discount > 0 else 1.0

    p = prescribe(mu[:lead] * factor,
                  sigma[:lead] * factor,
                  inv,
                  PolicyParams(service_z=Z_BY_SERVICE[service]),
                  sigma_lt=float(sigma_cum[lead - 1]) * factor)
    p.update({"lead_time": lead, "service_level": service,
              "discount_pct": round(discount, 1),
              "epsilon": round(eps, 3), "demand_factor": round(factor, 3)})
    return jsonify(p)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
