"""Station references and disturbances (docs/01_research_design.md; protocol section 4).

Disturbance = observed - reference, per station and component. References used so far:
- training median: median of each station and component over the training period;
- pre-event quiet constant: median over the quiet days before the event window (D14: UT days in
  the 20 days before the window with all eight Kp <= 2+), each station using only the days on
  which its X and Y are present in at least 90 % of the minutes (`quiet_constant`);
- decomposed (D03): CHAOS core field + station bias + quiet daily variation. The bias is
  piecewise constant; `best_step` finds a single breakpoint in a station's baseline series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440


def median_reference(data: pd.DataFrame) -> pd.Series:
    """Median of every (station, component) column over all rows of `data`."""
    return data.median(skipna=True)


def quiet_day_reference(data: pd.DataFrame, quiet_days) -> pd.Series:
    """Median over the minutes of the given quiet UT days."""
    if len(quiet_days) == 0:
        raise ValueError("no quiet day to build a reference from")
    day = data.index.normalize()
    rows = day.isin(pd.DatetimeIndex(quiet_days))
    return data.loc[rows].median(skipna=True)


def qualifying_quiet_data(
    data: pd.DataFrame, quiet_days, *, min_coverage: float = 0.9
) -> pd.DataFrame:
    """The rows of `data` on the quiet days, with a station's values set to NaN on each day its X
    and Y are both present in fewer than `min_coverage` of the minutes (D14)."""
    if len(quiet_days) == 0:
        raise ValueError("no quiet day to build a reference from")
    day = data.index.normalize()
    q = data.loc[day.isin(pd.DatetimeIndex(quiet_days))].copy()
    qday = q.index.normalize()
    for st in q.columns.get_level_values("station").unique():
        both = q[(st, "X")].notna() & q[(st, "Y")].notna()
        cov = both.groupby(qday).transform("sum") / MINUTES_PER_DAY
        q.loc[(cov < min_coverage).to_numpy(), st] = np.nan
    return q


def quiet_constant(
    data: pd.DataFrame, quiet_days, *, min_coverage: float = 0.9
) -> pd.Series:
    """Pre-event quiet constant (D14): median over the quiet days on which the station qualifies
    (X and Y both present in at least `min_coverage` of the minutes); NaN without such a day."""
    return qualifying_quiet_data(data, quiet_days, min_coverage=min_coverage).median(
        skipna=True
    )


def decomposed_reference(
    data: pd.DataFrame,
    quiet_days,
    index: pd.DatetimeIndex,
    core_change: pd.Series,
    *,
    min_coverage: float = 0.9,
    smooth_min: int = 61,
    min_variation_days: int = 3,
) -> pd.DataFrame:
    """Decomposed reference (D03) on `index`: quiet constant + main-field change from the quiet
    days to the event (`core_change`, per (station, component)) + quiet daily variation. Constant
    and variation use only the qualifying quiet days of each station. A station with fewer than
    `min_variation_days` qualifying days takes the median curve of the stations that have enough
    (zero if none has): one quiet day at auroral latitudes can carry a substorm of 100 nT."""
    q = qualifying_quiet_data(data, quiet_days, min_coverage=min_coverage)
    constant = q.median(skipna=True)
    shift = core_change.reindex(constant.index).fillna(0.0)
    variation = quiet_variation(q, quiet_days, smooth_min=smooth_min)
    stations = q.columns.get_level_values("station").unique()
    n_days = pd.Series(
        {
            st: int(q[(st, "X")].groupby(q.index.normalize()).count().gt(0).sum())
            for st in stations
        }
    )
    few = n_days.index[n_days < min_variation_days]
    enough = n_days.index[n_days >= min_variation_days]
    for comp in q.columns.get_level_values("component").unique():
        cols = [(st, comp) for st in enough]
        network = variation[cols].median(axis=1) if cols else 0.0
        for st in few:
            variation[(st, comp)] = network
    return reference_series(constant + shift, variation, index)


def disturbance(data: pd.DataFrame, ref) -> pd.DataFrame:
    """data - ref. `ref` is a Series over (station, component) columns or a DataFrame aligned
    with `data` (a time-varying reference)."""
    if isinstance(ref, pd.Series):
        return data - ref.reindex(data.columns)
    return data - ref.reindex(index=data.index, columns=data.columns)


def daily_medians(data: pd.DataFrame, *, min_coverage: float = 0.9) -> pd.DataFrame:
    """Median of each UT day per (station, component); NaN for a station whose X and Y are both
    present in fewer than `min_coverage` of the day's minutes."""
    day = data.index.normalize()
    med = data.groupby(day).median()
    stations = data.columns.get_level_values("station").unique()
    for st in stations:
        both = data[(st, "X")].notna() & data[(st, "Y")].notna()
        cov = both.groupby(day).sum() / MINUTES_PER_DAY
        med.loc[cov < min_coverage, st] = np.nan
    med.index.name = "day"
    return med


def best_step(values, *, min_segment: int = 3) -> dict:
    """Single breakpoint of a piecewise-constant model fitted by least squares.

    Returns k (first index after the step), the step (mean after - mean before), the fraction of
    variance the step explains, and z = |step| / (robust sigma * sqrt(1/k + 1/(n-k))), with sigma
    from the median absolute deviation of the residuals.
    """
    x = np.asarray(values, float)
    n = len(x)
    if n < 2 * min_segment:
        raise ValueError("series too short for a step")
    total = np.sum((x - x.mean()) ** 2)
    best = None
    for k in range(min_segment, n - min_segment + 1):
        a, b = x[:k], x[k:]
        sse = np.sum((a - a.mean()) ** 2) + np.sum((b - b.mean()) ** 2)
        if best is None or sse < best[1]:
            best = (k, sse)
    k, sse = best
    step = x[k:].mean() - x[:k].mean()
    resid = np.concatenate([x[:k] - x[:k].mean(), x[k:] - x[k:].mean()])
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    z = abs(step) / (sigma * np.sqrt(1 / k + 1 / (n - k))) if sigma > 0 else np.inf
    return {
        "k": int(k),
        "step": float(step),
        "explained": float(1 - sse / total) if total > 0 else 0.0,
        "z": float(z),
    }


def quiet_variation(
    data: pd.DataFrame, quiet_days, *, smooth_min: int = 61
) -> pd.DataFrame:
    """Quiet daily variation: for each minute of the UT day, the median over the quiet days of
    (value - that day's median), smoothed by a centred circular running mean of smooth_min
    minutes. Index 0..1439 (minute of day), columns as `data`."""
    day = data.index.normalize()
    rows = day.isin(pd.DatetimeIndex(quiet_days))
    q = data.loc[rows]
    dev = q - q.groupby(q.index.normalize()).transform("median")
    mod = dev.index.hour * 60 + dev.index.minute
    curve = dev.groupby(mod).median().reindex(range(MINUTES_PER_DAY))
    half = smooth_min // 2
    wrapped = pd.concat([curve.iloc[-half:], curve, curve.iloc[:half]])
    smooth = wrapped.rolling(smooth_min, center=True, min_periods=1).mean()
    out = smooth.iloc[half : half + MINUTES_PER_DAY]
    out.index = range(MINUTES_PER_DAY)
    out.index.name = "minute_of_day"
    return out


def reference_series(
    constant: pd.Series, variation: pd.DataFrame, index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Time-varying reference constant + variation[minute of day] on `index`."""
    mod = index.hour * 60 + index.minute
    var = variation.reindex(columns=constant.index).to_numpy()[mod]
    return pd.DataFrame(var + constant.to_numpy(), index=index, columns=constant.index)
