"""
rules.py - the uncertainty-aware prescriptive layer (build-order step 6).

This is the project's core novelty: the safety stock / reorder logic is driven by the
*model's own predicted uncertainty*, and the recommendation strength is modulated by how
confident that forecast is. It consumes the LSTM output - per-step mean `mu_t` plus the
lead-time demand std - and the current inventory, and emits a concrete action. It is a
pure function and does not care where the numbers came from (brief s7).

For a lead time of L weeks (mu, sigma have length L):
    expected lead-time demand   D_LT   = sum(mu_t)
    lead-time demand std        sig_LT = the MC sample-path-sum std passed in by the
                                         caller (preserves week-to-week correlation);
                                         falls back to sqrt(sum(sigma_t^2)) - the
                                         independence approximation - if not provided
    safety stock                SS     = Z * sig_LT      (Z = service-level factor)
    reorder point               ROP    = D_LT + SS
    order-up-to level           S      = mean_weekly*(L+R) + SS    (R = review period)
    relative uncertainty        CV     = sig_LT / D_LT   (drives confidence)

Decision:
    inventory <= ROP                  -> RESTOCK   (or REVIEW_RESTOCK if CV is high)
    plenty of cover even if demand is
    optimistically high               -> DISCOUNT  (or REVIEW_DISCOUNT if CV is high)
    otherwise                         -> ON_TRACK

The confidence gate (CV) is the novel twist: a very uncertain forecast never triggers an
aggressive automatic action - it is downgraded to a manual-review suggestion instead.

All thresholds are explicit, tunable, and must be justified in the thesis.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class PolicyParams:
    service_z: float = 1.65      # safety-stock service-level factor (~95%); 2.33 ~ 99%
    review_weeks: int = 2        # review period R the order-up-to level must also cover
    cv_low: float = 0.25         # CV <= cv_low  -> high confidence
    cv_high: float = 0.60        # CV >= cv_high -> low confidence (defer to manual review)
    overstock_weeks: float = 8.0 # > this many weeks of cover (even if demand is high) -> discount
    overstock_z: float = 1.28    # optimistic (~90th pct) demand quantile for the overstock test


def _confidence(cv: float, p: PolicyParams) -> str:
    if cv <= p.cv_low:
        return "high"
    if cv >= p.cv_high:
        return "low"
    return "medium"


def prescribe(mu, sigma, current_inventory: float, p: PolicyParams | None = None,
              sigma_lt: float | None = None) -> dict:
    """Turn an uncertain lead-time forecast + current stock into one prescription.

    sigma_lt: the lead-time-sum std from the MC sample paths (preferred). When None,
    falls back to the independence approximation sqrt(sum(sigma_t^2)) - which is also
    exactly the textbook sigma*sqrt(L) when a constant historical sigma is passed, so
    the classical policy in the backtest reuses this same function unchanged.
    """
    p = p or PolicyParams()
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    L = len(mu)

    d_lt = float(mu.sum())
    sig_lt = float(sigma_lt) if sigma_lt is not None else float(np.sqrt(np.sum(sigma ** 2)))
    ss = p.service_z * sig_lt
    rop = d_lt + ss
    mean_weekly = d_lt / L if L else 0.0
    order_up_to = mean_weekly * (L + p.review_weeks) + ss
    cv = sig_lt / d_lt if d_lt > 0 else math.inf
    confidence = _confidence(cv, p)

    # Weeks of cover if demand came in at the optimistic (upper-quantile) weekly rate.
    sigma_weekly = sig_lt / math.sqrt(L) if L else 0.0
    optimistic_weekly = mean_weekly + p.overstock_z * sigma_weekly
    weeks_cover_opt = current_inventory / optimistic_weekly if optimistic_weekly > 0 else math.inf

    # Decision -------------------------------------------------------------------------
    suggested = 0
    if current_inventory <= rop:                       # stockout risk
        suggested = max(0, int(math.ceil(order_up_to - current_inventory)))
        if confidence == "low":
            action = "REVIEW_RESTOCK"
            message = (f"Possible stockout (stock {round(current_inventory)} <= reorder "
                       f"point {round(rop)}) but the forecast is uncertain "
                       f"(CV {cv:.2f}) - review manually.")
        else:
            action = "RESTOCK"
            message = (f"Restock ~{suggested} units now: stock {round(current_inventory)} "
                       f"is at/below the reorder point {round(rop)} "
                       f"(lead-time demand {round(d_lt)} +/- {round(sig_lt)}).")
    elif weeks_cover_opt > p.overstock_weeks:          # overstock even under high demand
        if confidence == "low":
            action = "REVIEW_DISCOUNT"
            message = (f"Possible overstock (~{round(weeks_cover_opt)} weeks of cover) but the "
                       f"forecast is uncertain (CV {cv:.2f}) - review manually.")
        else:
            action = "DISCOUNT"
            message = (f"Consider a discount: ~{round(weeks_cover_opt)} weeks of cover remain "
                       f"even under optimistic demand.")
    else:
        action = "ON_TRACK"
        message = "On track: stock is adequate for the forecast lead-time demand."

    return {
        "current_inventory": round(current_inventory),
        "lead_time_demand": round(d_lt),
        "lead_time_demand_std": round(sig_lt),
        "safety_stock": round(ss),
        "reorder_point": round(rop),
        "order_up_to": round(order_up_to),
        "weeks_of_cover_opt": round(weeks_cover_opt, 1),
        "cv": round(cv, 2),
        "confidence": confidence,
        "action": action,
        "suggested_order_qty": suggested,
        "message": message,
    }
