"""Locked-event guard. T2 acceptance: reading a locked test event in default mode raises."""

import sys
import types

import pandas as pd
import pytest
from helpers import store_day

from p1.io import image, locked

ST = ["KEV", "SOD"]


def test_protected_periods_follow_config():
    periods = {eid: (lo, hi) for eid, lo, hi in locked.protected_periods()}
    assert len(periods) == 9
    # window 2024-05-10 .. 05-15 plus the 20-day reference period before it
    assert periods["20240510"] == (
        pd.Timestamp("2024-04-20"),
        pd.Timestamp("2024-05-15"),
    )


@pytest.mark.parametrize(
    "start, end",
    [
        ("2024-05-10", "2024-05-11"),  # inside the window
        (
            "2024-04-20 00:00",
            "2024-04-20 00:01",
        ),  # first minute of the reference period
        ("2024-05-14 23:59", "2024-05-15 00:00"),  # last minute of the window
        ("2024-03-01", "2024-06-01"),  # a range around the whole event
    ],
)
@pytest.mark.parametrize("validity_only", [False, True])
def test_default_mode_blocks_locked_periods(tmp_path, start, end, validity_only):
    with pytest.raises(locked.LockedEventError):
        image.read_range(start, end, ST, validity_only=validity_only, root=tmp_path)


def test_real_cached_test_data_is_refused():
    """The downloaded files of a locked event cannot be read through the loader."""
    stations = [
        "ABK",
        "AND",
        "IVA",
        "KEV",
        "KIL",
        "KIR",
        "MAS",
        "MUO",
        "OUJ",
        "PEL",
        "SOD",
        "TRO",
    ]
    with pytest.raises(locked.LockedEventError):
        image.read_range("2015-03-17", "2015-03-18", stations, validity_only=True)


def test_guard_runs_before_files_are_opened(tmp_path):
    store_day(tmp_path, "2024-05-10", ST)
    with pytest.raises(locked.LockedEventError):
        image.read_range("2024-05-10", "2024-05-11", ST, root=tmp_path)


@pytest.mark.parametrize("day", ["2024-04-19", "2024-05-15"])
def test_days_next_to_a_locked_event_are_open(tmp_path, day):
    store_day(tmp_path, day, ST)
    end = pd.Timestamp(day) + pd.Timedelta(days=1)
    assert len(image.read_range(day, end, ST, root=tmp_path)) == 1440


def test_locked_access_refused_outside_the_runner():
    with pytest.raises(locked.LockedEventError), locked.locked_access():
        pass


def test_runner_identity(monkeypatch):
    fake = types.ModuleType("__main__")
    fake.__file__ = str(locked.RUNNER)
    monkeypatch.setitem(sys.modules, "__main__", fake)
    assert locked._running_runner()
    fake.__file__ = str(locked.RUNNER.with_name("screen_test_events.py"))
    assert not locked._running_runner()


def test_locked_access_opens_only_inside_the_block(tmp_path, monkeypatch):
    store_day(tmp_path, "2024-05-10", ST)
    monkeypatch.setattr(locked, "_running_runner", lambda: True)
    with locked.locked_access():
        assert (
            len(image.read_range("2024-05-10", "2024-05-11", ST, root=tmp_path)) == 1440
        )
    with pytest.raises(locked.LockedEventError):
        image.read_range("2024-05-10", "2024-05-11", ST, root=tmp_path)
    # the guard closes again after an error inside the block
    with pytest.raises(RuntimeError), locked.locked_access():
        raise RuntimeError("evaluation failed")
    with pytest.raises(locked.LockedEventError):
        image.read_range("2024-05-10", "2024-05-11", ST, root=tmp_path)


def test_runner_needs_a_signoff_for_the_locked_run():
    import subprocess
    import sys

    from p1 import paths

    r = subprocess.run(
        [sys.executable, str(paths.ROOT / "scripts" / "run_locked_test.py")],
        capture_output=True,
        text=True,
        cwd=paths.ROOT,
        check=False,
    )
    assert r.returncode != 0
    assert "--pi-signoff" in r.stderr
