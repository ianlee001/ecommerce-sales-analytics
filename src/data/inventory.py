"""
inventory.py - Simulated inventory trajectory (a DOCUMENTED ASSUMPTION).

Online Retail II has no stock data, so we synthesise a plausible inventory path per SKU
from *realised* weekly sales using a textbook periodic-review (s, S) policy. This is NOT a
forecast and NOT the prescriptive layer - it only gives the `inventory` column of the
standard format something realistic to be. Every parameter is explicit and tunable and
must be listed as an assumption in the thesis.

Event order within a week t:
  1. receive any order scheduled to arrive at week t,
  2. record stock at the START of the bucket  (this is the `inventory` column),
  3. subtract that week's realised demand      (clamped at 0 - no backorders modelled),
  4. review: if on-hand <= reorder point s and no order is already in transit, place an
     order up to S that arrives after `lead_time` weeks.
"""
from __future__ import annotations

import numpy as np


def simulate_inventory(demand, lead_time: int = 1, review_weeks: int = 2,
                       service_z: float = 1.65):
    """Return (inventory_at_start_of_each_week, params_used).

    Parameters
    ----------
    demand        : 1-D array of realised weekly units sold for one SKU.
    lead_time     : weeks between placing and receiving an order (assumption).
    review_weeks  : periodic-review window the order-up-to level must cover.
    service_z     : service-level factor for the baseline safety stock (1.65 ~= 95%).
    """
    demand = np.asarray(demand, dtype=float)
    mu = float(demand.mean())
    sigma = float(demand.std(ddof=0))
    L, R = int(lead_time), int(review_weeks)

    # Textbook levels (documented assumption, not learned):
    #   safety stock covers demand variability over the lead time,
    #   reorder point s = expected lead-time demand + safety stock,
    #   order-up-to  S = expected demand over (lead time + review) + safety.
    safety = service_z * sigma * np.sqrt(L)
    s = mu * L + safety
    S = mu * (L + R) + service_z * sigma * np.sqrt(L + R)
    S = max(S, s + mu)            # guarantee a sane gap S > s

    inv = S                       # start the simulation full
    pipeline: dict[int, float] = {}   # week_of_arrival -> quantity in transit
    starts = np.empty(len(demand), dtype=float)

    for t in range(len(demand)):
        inv += pipeline.pop(t, 0.0)          # 1. receipts
        starts[t] = inv                      # 2. stock at start of bucket
        inv = max(inv - demand[t], 0.0)      # 3. demand (no backorders)
        if inv <= s and not pipeline:        # 4. review -> order up to S
            pipeline[t + L] = S - inv

    params = dict(mu=mu, sigma=sigma, reorder_point_s=float(s), order_up_to_S=float(S),
                  safety_stock=float(safety), lead_time=L, review_weeks=R,
                  service_z=service_z)
    return starts, params
