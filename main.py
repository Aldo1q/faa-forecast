#!/usr/bin/env python3
"""Forecaster for make.fun's "Forecast Polymarket's FAA outage attribution
prices minute by minute, 4 hours ahead" bounty.

Reads one case at a time (/input/<id>.in) and writes, for each market and
each of the 240 minutes after the checkpoint, five quantiles (5%, 25%, 50%,
75%, 95%) of that minute's Yes price, to /output/<id>.out.

Method: a random-walk forecast with a per-market, per-quantile spread.
Prices are mostly flat and then jump, so the distribution is heavy-tailed:
the inner quantiles (25%, 75%) use a robust small spread from the typical
minute-to-minute move, while the outer quantiles (5%, 95%) use a wider
spread from the size of the larger moves, so jumps are covered. The median
follows a heavily damped recent trend. The spread widens like sqrt(h) with
the horizon h, as a random walk's does. Volatilities are estimated from the
recent window of each market's own history, so quiet markets get tight
forecasts and volatile ones get wide ones -- unlike the house baseline,
which uses one fixed spread (0.0025) for every market.

Deterministic (no randomness, no network, standard library only).
"""

import json
import math
import os
import statistics

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

# --- tuning knobs (tuned on the practice cases; the practice loss is
# 0.286 vs the house baseline's 0.364) ---
WINDOW = 90            # minutes of recent history used for volatility
DRIFT_LOOKBACK = 60    # minutes over which the recent trend is measured
DRIFT_DAMP = 0.0       # how much of the recent trend the median follows
                        # (0: the median stays at the last price -- a trend
                        #  hurt the practice score, so no drift)
DRIFT_CAP = 0.001      # max |drift| per minute, in price units
SIGMA_INNER_FLOOR = 0.0003
SIGMA_INNER_CAP = 0.010
SIGMA_OUTER_FLOOR = 0.001
SIGMA_OUTER_CAP = 0.004
INNER_MULT = 1.0        # inner spread = median |1-min change| (not 1.4826x)
OUTER_MULT = 0.3        # weight of the std-based term in the outer spread


def clamp01(x):
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def price6(x):
    # clamp to [0,1] and round to 6 decimals (0, never -0)
    return round(clamp01(x) * 1e6) / 1e6 + 0.0


def _quantile(sorted_vals, q):
    """Linear-interpolated quantile of a sorted list (q in [0,1])."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def estimate_sigmas(prices):
    """Return (sigma_inner, sigma_outer) per-minute spreads for one market,
    estimated from its recent 1-minute changes."""
    n = len(prices)
    if n < 2:
        return SIGMA_INNER_FLOOR, SIGMA_OUTER_FLOOR
    w = min(WINDOW, n - 1)
    changes = [prices[n - 1 - i] - prices[n - 2 - i] for i in range(w)]
    abs_ch = sorted(abs(c) for c in changes)
    # inner: typical move -- robust (median absolute change). The 1.4826
    # normal-consistency factor is NOT applied: the median |change| itself
    # scores better on the practice (the moves are heavy-tailed).
    med_abs = _quantile(abs_ch, 0.5)
    sigma_inner = INNER_MULT * med_abs
    # If the market never moves in the window, fall back to a small floor so
    # the forecast is not degenerate; a truly flat market stays tight.
    sigma_inner = min(max(sigma_inner, SIGMA_INNER_FLOOR), SIGMA_INNER_CAP)
    # outer: size of the larger moves -- 95th percentile of |change|, scaled
    # to a Gaussian sigma (a 95% two-sided normal quantile is 1.96 sigma).
    p95 = _quantile(abs_ch, 0.95)
    sigma_outer = p95 / 1.96 if p95 > 0 else SIGMA_OUTER_FLOOR
    # Also respect the ordinary std (jumps inflate it) as a lower bound-ish
    # signal: take the larger of the two robust estimates, capped.
    if len(changes) >= 10:
        mean = sum(changes) / len(changes)
        var = sum((c - mean) ** 2 for c in changes) / len(changes)
        sigma_outer = max(sigma_outer, math.sqrt(var) * OUTER_MULT)
    sigma_outer = min(max(sigma_outer, SIGMA_OUTER_FLOOR), SIGMA_OUTER_CAP)
    # The outer spread must be at least the inner one.
    sigma_outer = max(sigma_outer, sigma_inner)
    return sigma_inner, sigma_outer


def estimate_drift(prices):
    """Heavily damped recent trend, per minute, in price units."""
    n = len(prices)
    if n < 2:
        return 0.0
    k = min(DRIFT_LOOKBACK, n - 1)
    drift = (prices[-1] - prices[-1 - k]) / k * DRIFT_DAMP
    if drift > DRIFT_CAP:
        drift = DRIFT_CAP
    elif drift < -DRIFT_CAP:
        drift = -DRIFT_CAP
    return drift


def forecast_market(prices, horizon):
    x = prices[-1]
    sigma_inner, sigma_outer = estimate_sigmas(prices)
    drift = estimate_drift(prices)
    rows = []
    for h in range(1, horizon + 1):
        root_h = math.sqrt(h)
        center = x + drift * h
        row = [
            price6(center + Z[0] * sigma_outer * root_h),
            price6(center + Z[1] * sigma_inner * root_h),
            price6(center),
            price6(center + Z[3] * sigma_inner * root_h),
            price6(center + Z[4] * sigma_outer * root_h),
        ]
        # guarantee non-decreasing (clamping can tie neighbours; never invert)
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in inputs:
        case_id = fname[: -len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = int(case.get("horizon_minutes", 240))
        forecasts = []
        for m in case["markets"]:
            prices = m["prices"]
            if not prices:
                raise ValueError("market %s has no prices" % m["id"])
            forecasts.append({"id": m["id"], "q": forecast_market(prices, horizon)})
        out = {"forecasts": forecasts}
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as f:
            f.write(json.dumps(out))
            f.write("\n")


if __name__ == "__main__":
    main()
