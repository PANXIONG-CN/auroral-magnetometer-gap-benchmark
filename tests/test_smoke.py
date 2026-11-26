import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

import p1

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))


def test_package_imports():
    assert p1.__version__


def test_configs_parse():
    for name in ["events.yaml", "stations.yaml", "defaults.yaml"]:
        assert load(name)


def test_test_events_are_locked():
    """T1 locked the test events (D12, D14). Each window plus its reference period stays clear
    of the training, validation and development periods (development with its reference period)."""
    events = load("events.yaml")
    lookback = pd.Timedelta(
        days=load("defaults.yaml")["event_screening"]["lookback_days"]
    )
    test = events["test"]
    ids = [e["id"] for e in test]
    assert len(test) >= 8 and len(set(ids)) == len(ids)
    assert all(e["locked"] is True for e in test)
    used = [
        (pd.Timestamp(events[g]["start"]), pd.Timestamp(events[g]["end"]))
        for g in ("training", "validation")
    ]
    used += [
        (pd.Timestamp(e["start"]) - lookback, pd.Timestamp(e["end"]))
        for e in events["development"]
    ]
    screened = {c["id"]: c for c in events["test_candidates"]}
    for e in test:
        start, end = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
        assert start == start.normalize() and end - start == pd.Timedelta(days=5)
        c = screened[e["id"]]  # the window and verdict recorded by the T1 screening
        assert (c["start"], c["end"], c["rec"]) == (e["start"], e["end"], "pass")
        for lo, hi in used:
            assert end <= lo or hi <= start - lookback, (e["id"], lo, hi)


def test_locked_list_unchanged_since_lock():
    """The test list in the working tree equals the one in the lock commit (D12)."""
    events = load("events.yaml")
    commit = events["test_locked_commit"]
    assert isinstance(commit, str) and len(commit) == 40
    try:
        old = subprocess.run(
            ["git", "show", f"{commit}:config/events.yaml"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history with the lock commit is not available")
    assert yaml.safe_load(old)["test"] == events["test"]
