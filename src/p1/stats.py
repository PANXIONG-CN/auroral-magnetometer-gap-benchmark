"""Event-level aggregation and bootstrap (docs/03_evaluation_protocol.md section 6; hard rule 5).

The 3-hour scenarios of one event are not independent samples. Scenario metrics are first combined
into one value per event, and confidence intervals resample events.

Event-level values (D15, pending PI confirmation):
- rmse_h, rmse_x, rmse_y: pooled over scored minutes, sqrt(sum n_i rmse_i^2 / sum n_i), which
  equals the RMSE over all scored minutes of the event; dbdt_rmse likewise with n_dbdt.
- peak_err: mean over scenarios with scored minutes; peak_dt_abs_min: mean |peak_dt_min|.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

POOLED = {
    "rmse_h": "n_scored",
    "rmse_x": "n_scored",
    "rmse_y": "n_scored",
    "dbdt_rmse": "n_dbdt",
}


def event_summary(scenarios: pd.DataFrame, by=("event", "method")) -> pd.DataFrame:
    """One row per group in `by` from scenario-level metrics (columns as p1.metrics.score)."""
    by = list(by)
    rows = []
    for key, g in scenarios.groupby(by, sort=True):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        g = g[g["n_scored"] > 0]
        row["n_scenarios"] = len(g)
        row["n_scored"] = int(g["n_scored"].sum())
        for metric, weight in POOLED.items():
            w = g[weight].to_numpy(float)
            v = g[metric].to_numpy(float)
            use = w > 0
            row[metric] = (
                float(np.sqrt(np.sum(w[use] * v[use] ** 2) / np.sum(w[use])))
                if use.any()
                else float("nan")
            )
        row["peak_err"] = float(g["peak_err"].mean()) if len(g) else float("nan")
        row["peak_dt_abs_min"] = (
            float(g["peak_dt_min"].abs().mean()) if len(g) else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean(values, *, n_resamples=None, ci=None, seed=None) -> dict:
    """Mean of event-level values with a percentile bootstrap interval over events.

    Defaults come from config/defaults.yaml (bootstrap). NaN values are dropped.
    """
    cfg = config.load("defaults")["bootstrap"]
    n_resamples = n_resamples or cfg["n_resamples"]
    ci = ci or cfg["ci"]
    seed = cfg["seed"] if seed is None else seed
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_resamples, v.size))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": int(v.size)}


def paired_difference(
    events: pd.DataFrame, metric: str, a: str, b: str, *, key="method", on="event", **kw
) -> dict:
    """Event-level difference a - b of `metric`, bootstrapped over events present for both.

    `events` must hold one row per (on, key), e.g. from event_summary on a single station set,
    reference and mask (hard rule 4); duplicates raise.
    """
    wide = events.pivot(index=on, columns=key, values=metric)
    both = wide[[a, b]].dropna()
    diff = both[a] - both[b]
    out = bootstrap_mean(diff.to_numpy(), **kw)
    out["per_event"] = diff
    return out
