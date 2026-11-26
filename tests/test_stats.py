"""Event-level aggregation and bootstrap."""

import math

import numpy as np
import pandas as pd
import pytest

from p1 import metrics, stats

NAN = float("nan")


def scenario_rows():
    cols = [
        "event",
        "method",
        "n_scored",
        "rmse_h",
        "rmse_x",
        "rmse_y",
        "peak_err",
        "peak_dt_min",
        "n_dbdt",
        "dbdt_rmse",
    ]
    unscored = ["E1", "secs", 0, NAN, NAN, NAN, NAN, NAN, 0, NAN]  # target missing
    rows = [
        ["E1", "secs", 4, 3.0, 2.0, 1.0, 10.0, -4.0, 3, 2.0],
        ["E1", "secs", 12, 4.0, 3.0, 2.0, 20.0, 2.0, 11, 1.0],
        unscored,
    ]
    return pd.DataFrame(rows, columns=cols)


def test_event_summary_pools_minutes():
    s = stats.event_summary(scenario_rows()).iloc[0]
    assert (s.event, s.method, s.n_scenarios, s.n_scored) == ("E1", "secs", 2, 16)
    assert s.rmse_h == pytest.approx(math.sqrt((4 * 9 + 12 * 16) / 16))
    assert s.rmse_x == pytest.approx(math.sqrt((4 * 4 + 12 * 9) / 16))
    assert s.rmse_y == pytest.approx(math.sqrt((4 * 1 + 12 * 4) / 16))
    assert s.dbdt_rmse == pytest.approx(math.sqrt((3 * 4 + 11 * 1) / 14))
    assert s.peak_err == pytest.approx(15.0) and s.peak_dt_abs_min == pytest.approx(3.0)


def test_pooled_rmse_equals_rmse_over_all_minutes():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-05-10", periods=360, freq="min")
    truth = pd.DataFrame(rng.normal(size=(360, 2)) * 50, index=idx, columns=["X", "Y"])
    pred = truth + rng.normal(size=(360, 2)) * 10
    pred.iloc[200:230] = np.nan
    rows = [
        dict(
            event="E1",
            method="m",
            **metrics.score(pred.iloc[i : i + 180], truth.iloc[i : i + 180]),
        )
        for i in (0, 180)
    ]
    pooled = stats.event_summary(pd.DataFrame(rows)).iloc[0].rmse_h
    assert pooled == pytest.approx(metrics.score(pred, truth)["rmse_h"])


def test_bootstrap_basic_cases():
    assert stats.bootstrap_mean([2.0, 2.0, 2.0], n_resamples=500) == {
        "mean": 2.0,
        "lo": 2.0,
        "hi": 2.0,
        "n": 3,
    }
    r = stats.bootstrap_mean([0.0, 1.0], n_resamples=20000, seed=1)
    assert (r["mean"], r["lo"], r["hi"]) == (0.5, 0.0, 1.0)
    assert stats.bootstrap_mean([1.0, NAN, 3.0], n_resamples=100)["n"] == 2
    empty = stats.bootstrap_mean([NAN])
    assert empty["n"] == 0 and math.isnan(empty["mean"])


def test_bootstrap_is_reproducible_and_bounded():
    v = [3.0, 7.0, 4.0, 10.0, 5.0, 6.0, 2.0, 8.0, 9.0]
    a = stats.bootstrap_mean(v, n_resamples=2000, seed=7)
    assert a == stats.bootstrap_mean(v, n_resamples=2000, seed=7)
    assert min(v) <= a["lo"] < a["mean"] < a["hi"] <= max(v)
    assert a["mean"] == pytest.approx(np.mean(v))


def test_bootstrap_defaults_come_from_config():
    v = [1.0, 2.0, 4.0]
    assert stats.bootstrap_mean(v) == stats.bootstrap_mean(
        v, n_resamples=10000, ci=0.95, seed=0
    )


def test_paired_difference_by_event():
    ev = pd.DataFrame(
        {
            "event": ["E1", "E2", "E3", "E1", "E2", "E3", "E4"],
            "method": ["idw"] * 3 + ["secs"] * 4,
            "rmse_h": [5.0, 6.0, 7.0, 4.0, 4.0, 4.0, 1.0],
        }
    )
    d = stats.paired_difference(ev, "rmse_h", "idw", "secs", n_resamples=2000)
    assert d["n"] == 3 and d["mean"] == pytest.approx(2.0)
    assert d["per_event"].to_dict() == {"E1": 1.0, "E2": 2.0, "E3": 3.0}
    with pytest.raises(ValueError):  # one row per (event, method) only
        stats.paired_difference(pd.concat([ev, ev]), "rmse_h", "idw", "secs")
