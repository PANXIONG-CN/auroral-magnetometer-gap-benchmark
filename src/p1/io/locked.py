"""Guard for the locked test events (CLAUDE.md hard rule 1; docs/05_decisions.md D12, D14).

Every test event with `locked: true` in config/events.yaml is protected together with its
reference period, the `event_screening.lookback_days` days before the window. Reading any minute
of a protected period raises LockedEventError. Only scripts/run_locked_test.py can open them,
inside `locked_access()`.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .. import config, paths

RUNNER = paths.ROOT / "scripts" / "run_locked_test.py"
_open = False


class LockedEventError(PermissionError):
    """A read touched a locked test event outside scripts/run_locked_test.py."""


def protected_periods() -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """(event id, start of reference period, window end) for each locked test event."""
    events = config.load("events")
    days = config.load("defaults")["event_screening"]["lookback_days"]
    lookback = pd.Timedelta(days=days)
    return [
        (str(e["id"]), pd.Timestamp(e["start"]) - lookback, pd.Timestamp(e["end"]))
        for e in events.get("test") or []
        if e.get("locked")
    ]


def check(start, end) -> None:
    """Raise LockedEventError if [start, end) overlaps a protected period, unless opened."""
    if _open:
        return
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    hits = [eid for eid, lo, hi in protected_periods() if start < hi and lo < end]
    if hits:
        raise LockedEventError(
            f"[{start}, {end}) overlaps locked test event(s) {', '.join(hits)} "
            "(window or reference period); only scripts/run_locked_test.py may read them"
        )


def _running_runner() -> bool:
    main = getattr(sys.modules.get("__main__"), "__file__", None)
    return bool(main) and Path(main).resolve() == RUNNER.resolve()


@contextmanager
def locked_access():
    """Open the locked test events for the duration of the block.

    Raises LockedEventError unless the running program is scripts/run_locked_test.py.
    """
    global _open
    if not _running_runner():
        raise LockedEventError(
            "locked test events can only be opened by scripts/run_locked_test.py"
        )
    previous, _open = _open, True
    try:
        yield
    finally:
        _open = previous
