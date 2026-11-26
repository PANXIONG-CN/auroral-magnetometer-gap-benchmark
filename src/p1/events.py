"""Event periods from config/events.yaml and candidate test-event windows.

T2 adds the locked-event guard on top of this module.
"""

from __future__ import annotations

import pandas as pd

from . import config, storms

DAY = pd.Timedelta(days=1)


def screening_cfg() -> dict:
    return config.load("defaults")["event_screening"]


def fixed_periods(group: str) -> list[dict]:
    """[{id, start, end}] for 'training', 'validation', 'development' or 'test'."""
    ev = config.load("events")
    if group in ("training", "validation"):
        p = ev[group]
        return [
            {
                "id": group,
                "start": pd.Timestamp(p["start"]),
                "end": pd.Timestamp(p["end"]),
            }
        ]
    if group in ("development", "test"):
        return [
            {
                "id": str(e["id"]),
                "start": pd.Timestamp(e["start"]),
                "end": pd.Timestamp(e["end"]),
            }
            for e in ev[group] or []
        ]
    raise ValueError(f"unknown group {group!r}")


def candidates() -> list[dict]:
    return [dict(c, id=str(c["id"])) for c in config.load("events")["test_candidates"]]


def candidate_windows(dst: pd.Series, cfg: dict | None = None) -> list[dict]:
    """For each test candidate: Dst minimum, main-phase onset and the proposed window."""
    cfg = cfg or screening_cfg()
    out = []
    for c in candidates():
        mp = storms.main_phase(
            dst,
            c["storm_date"],
            search_days=tuple(cfg["dst_search_days"]),
            onset_lookback_h=cfg["onset_lookback_h"],
        )
        start, end = storms.window_from_onset(mp["t_onset"], cfg["window_days"])
        out.append(
            {
                "id": c["id"],
                "storm_date": pd.Timestamp(c["storm_date"]),
                **mp,
                "start": start,
                "end": end,
            }
        )
    return out


def dst_span_for_candidates(
    cfg: dict | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Time spans whose Dst is needed to place the candidate windows."""
    cfg = cfg or screening_cfg()
    lo, hi = cfg["dst_search_days"]
    spans = []
    for c in candidates():
        d0 = pd.Timestamp(c["storm_date"])
        spans.append(
            (
                d0 + lo * DAY - cfg["onset_lookback_h"] * pd.Timedelta(hours=1),
                d0 + hi * DAY,
            )
        )
    return spans


def days_between(start, end) -> list[pd.Timestamp]:
    """UT days touched by [start, end)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    return list(
        pd.date_range(
            start.normalize(), (end - pd.Timedelta(minutes=1)).normalize(), freq="D"
        )
    )
