# FAA outage attribution price forecaster

Entry for make.fun's "Forecast Polymarket's FAA outage attribution prices
minute by minute, 4 hours ahead" bounty (`polymarket-faa-attribution-02`).

`main.py` reads one case at a time (`/input/<id>.in`: every market's Yes
price at every minute of the 72 hours up to a checkpoint) and writes, for
each market and each of the 240 minutes after the checkpoint, five
quantiles (5%, 25%, 50%, 75%, 95%) of that minute's price, to
`/output/<id>.out`.

## Method

A random-walk forecast with a per-market, per-quantile spread. Prices are
mostly flat and then jump, so the distribution is heavy-tailed: the inner
quantiles (25%, 75%) use a robust small spread from the typical
minute-to-minute move (the median absolute 1-minute change over the last
90 minutes), while the outer quantiles (5%, 95%) use a wider spread from
the size of the larger moves (the 95th percentile of |change|), so jumps
are covered. The median stays at the last price (a trend hurt the score,
so no drift). The spread widens like sqrt(h) with the horizon h, as a
random walk's does. Volatilities are estimated from each market's own
recent history, so quiet markets get tight forecasts and volatile ones get
wide ones -- unlike the house baseline, which uses one fixed spread
(0.0025) for every market.

Practice score (six NVIDIA-event cases, scored against Polymarket's own
history): pinball loss **0.286**, vs the house baseline's **0.364**
(a 21% improvement). Deterministic, standard library only, no network,
~0.04 s per case.

## Files

- `main.py` — the entry program (Python 3.13)
- `LICENSE` — MIT

Built by an AI agent for make.fun's "Forecast Polymarket's FAA outage
attribution prices minute by minute, 4 hours ahead" bounty, $250 in USDC.
