"""Scenarios and hidden-station isolation (docs/03_evaluation_protocol.md sections 3 and 4).

A scenario hides one target station for one block of an event window (180 minutes; 40 blocks
per 5-day window). `split` divides the event data into what a reconstruction method may see and
what only the scoring function sees (CLAUDE.md hard rule 2):

- ScenarioInputs.current: neighbour stations in [start, end). The target's columns are absent.
- ScenarioInputs.history: neighbours and target in [start - history_min, start), for temporal
  methods. Spatial methods use `current` only.
- Truth.values: the target's components in [start, end). It goes to p1.metrics.score only.

Both objects hold copies, so a method cannot change the truth or another scenario's data. The
data passed to `split` should come from p1.io.image.read_range, which applies the locked-event
guard; for disturbances, the target's reference must come from data before the event window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io.image import COMPONENTS
from .io.kp import kp_thirds

MINUTE = pd.Timedelta(minutes=1)
KP_INTERVAL = pd.Timedelta(hours=3)


@dataclass(frozen=True)
class Scenario:
    event: str
    target: str
    start: pd.Timestamp
    end: pd.Timestamp
    neighbours: tuple[str, ...]
    kp: float = float("nan")  # highest Kp of the 3-hour intervals the scenario overlaps


@dataclass(frozen=True)
class ScenarioInputs:
    """What a method may see. Contains no value of the target inside [start, end)."""

    scenario: Scenario
    current: pd.DataFrame
    history: pd.DataFrame


@dataclass(frozen=True)
class Truth:
    """The hidden target values; for p1.metrics.score only."""

    scenario: Scenario
    values: pd.DataFrame  # columns X, Y, Z; index = every minute of the scenario


def _max_kp(kp3h: pd.DataFrame, start, end) -> float:
    t = kp3h.index
    seg = kp3h.loc[(t < end) & (t + KP_INTERVAL > start), "kp"]
    return float(seg.max()) if len(seg) else float("nan")


def make_scenarios(
    event,
    start,
    end,
    stations,
    *,
    targets=None,
    length_min: int = 180,
    kp3h: pd.DataFrame | None = None,
) -> list[Scenario]:
    """Non-overlapping blocks of length_min for every target; the other stations are neighbours.

    targets defaults to every station (leave-one-out). [start, end) must split into whole blocks.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    total = (end - start) / MINUTE
    if total <= 0 or total % length_min:
        raise ValueError(
            f"[{start}, {end}) does not split into {length_min}-minute blocks"
        )
    stations = list(stations)
    targets = stations if targets is None else list(targets)
    unknown = sorted(set(targets) - set(stations))
    if unknown:
        raise ValueError(f"targets not in the station set: {unknown}")
    block = length_min * MINUTE
    out = []
    for target in targets:
        neighbours = tuple(s for s in stations if s != target)
        for i in range(int(total // length_min)):
            s = start + i * block
            kp = _max_kp(kp3h, s, s + block) if kp3h is not None else float("nan")
            out.append(Scenario(str(event), target, s, s + block, neighbours, kp))
    return out


def kp_group(kp: float, threshold: float = 5) -> str:
    """'Kp>=5' or 'Kp<5' (by Kp thirds, so 5- counts as below 5); 'unknown' for NaN."""
    if np.isnan(kp):
        return "unknown"
    above = kp_thirds(kp) >= kp_thirds(threshold)
    return f"Kp>={threshold:g}" if above else f"Kp<{threshold:g}"


def split(
    scenario: Scenario, data: pd.DataFrame, *, history_min: int
) -> tuple[ScenarioInputs, Truth]:
    """Split event data (index: minutes; columns: (station, component)) for one scenario."""
    if data.columns.nlevels != 2:
        raise ValueError("data columns must be (station, component)")
    minutes = pd.date_range(
        scenario.start, scenario.end - MINUTE, freq="min", name="time"
    )
    past = pd.date_range(
        scenario.start - history_min * MINUTE,
        periods=history_min,
        freq="min",
        name="time",
    )

    def cols(stations):
        return pd.MultiIndex.from_product(
            [list(stations), COMPONENTS], names=["station", "component"]
        )

    current = data.reindex(index=minutes, columns=cols(scenario.neighbours)).copy()
    history = data.reindex(
        index=past, columns=cols((*scenario.neighbours, scenario.target))
    ).copy()
    truth = data.reindex(index=minutes, columns=cols([scenario.target])).copy()
    truth.columns = truth.columns.droplevel("station")
    return ScenarioInputs(scenario, current, history), Truth(scenario, truth)
