"""Storm verification and test-event screening (T1; docs/03_evaluation_protocol.md section 2).

Definitions (confirmed by the PI on 2026-11-17, docs/05_decisions.md D14):
- Dst minimum: lowest hourly Dst within [storm_date - 1 d, storm_date + 2 d).
- Main-phase onset: the hour of maximum Dst within the 24 h before the Dst minimum.
- Event window: 5 days starting 00:00 UT on the day of the main-phase onset.
- Quiet day: a UT day within the 20 days before the window whose eight Kp values are all present
  and at most 2+.
- Station coverage: fraction of minutes in which both X and Y are present. A station has a
  reference if it covers >= 90 % of some quiet day.
- Station rule, minute by minute: among stations with a reference, >= 9 valid in >= 99 % of the
  window minutes.
These functions only read availability masks of magnetometer data, never values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io.kp import kp_thirds

HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)


def _between(s, lo, hi):
    return s[(s.index >= lo) & (s.index < hi)]


def main_phase(
    dst: pd.Series, storm_date, *, search_days=(-1, 2), onset_lookback_h: int = 24
) -> dict:
    """Dst minimum near storm_date and the main-phase onset before it."""
    d0 = pd.Timestamp(storm_date)
    seg = _between(dst, d0 + search_days[0] * DAY, d0 + search_days[1] * DAY).dropna()
    if seg.empty:
        raise ValueError(f"no Dst around {d0.date()}")
    t_min = seg.idxmin()
    pre = _between(dst, t_min - onset_lookback_h * HOUR, t_min + HOUR).dropna()
    t_onset = pre.idxmax()
    return {
        "dst_min": float(seg.min()),
        "t_dst_min": t_min,
        "t_onset": t_onset,
        "dst_at_onset": float(pre.max()),
        "main_phase_h": int((t_min - t_onset) / HOUR),
    }


def window_from_onset(t_onset, days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(t_onset).normalize()
    return start, start + days * DAY


def kp_summary(kp3h: pd.DataFrame, start, end, *, active_kp: float = 5.0) -> dict:
    """Kp maximum, its interval, hours with Kp >= active_kp, and whether all values are definitive."""
    seg = _between(kp3h, pd.Timestamp(start), pd.Timestamp(end))
    thirds = seg["kp"].dropna().map(kp_thirds)
    return {
        "kp_max": float(seg["kp"].max()),
        "t_kp_max": seg["kp"].idxmax(),
        "hours_kp_active": int((thirds >= kp_thirds(active_kp)).sum() * 3),
        "kp_definitive": bool(seg["definitive"].all()),
    }


def quiet_days(
    kp3h: pd.DataFrame, before, lookback_days: int, *, kp_max="2+"
) -> list[pd.Timestamp]:
    """UT days in [before - lookback_days, before) with all eight Kp present and <= kp_max."""
    before = pd.Timestamp(before)
    seg = _between(kp3h, before - lookback_days * DAY, before)
    limit = kp_thirds(kp_max)
    out = []
    for day, g in seg.groupby(seg.index.normalize()):
        k = g["kp"]
        if len(k) == 8 and k.notna().all() and (k.map(kp_thirds) <= limit).all():
            out.append(day)
    return out


def coverage(valid: pd.DataFrame, start, end, stations) -> pd.Series:
    """Per station: fraction of minutes in [start, end) with both X and Y present.

    `valid` is a boolean availability mask with columns (station, component).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    seg = valid[(valid.index >= start) & (valid.index < end)]
    expected = int((end - start) / pd.Timedelta(minutes=1))
    out = {}
    for st in stations:
        if (st, "X") in seg.columns and (st, "Y") in seg.columns:
            out[st] = float((seg[(st, "X")] & seg[(st, "Y")]).sum()) / expected
        else:
            out[st] = 0.0
    return pd.Series(out, name="coverage")


def valid_station_counts(valid: pd.DataFrame, start, end, stations) -> np.ndarray:
    """Number of stations with both X and Y present, for each minute in [start, end)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    seg = valid[(valid.index >= start) & (valid.index < end)]
    both = [
        seg[(st, "X")] & seg[(st, "Y")] for st in stations if (st, "X") in seg.columns
    ]
    return np.sum(both, axis=0) if both else np.zeros(len(seg), dtype=int)


def quiet_day_coverage(valid: pd.DataFrame, days, stations) -> pd.Series:
    """Best coverage per station over the given quiet days; NaN when there are none."""
    if not days:
        return pd.Series(np.nan, index=list(stations), name="coverage")
    per_day = pd.concat([coverage(valid, d, d + DAY, stations) for d in days], axis=1)
    return per_day.max(axis=1).rename("coverage")


def per_minute_rule(
    valid: pd.DataFrame, start, end, stations, *, min_valid: int, min_frac: float
) -> dict:
    """Minute-by-minute station rule over [start, end): share of minutes with at least
    min_valid of the given stations valid, and whether it reaches min_frac."""
    counts = valid_station_counts(valid, start, end, stations)
    frac = float((counts >= min_valid).mean()) if len(counts) else 0.0
    return {
        "min_valid_per_minute": int(counts.min()) if len(counts) else 0,
        "frac_minutes_ok": frac,
        "ok": frac >= min_frac,
    }
