"""Metrics on hand-computed examples (T2 acceptance)."""

import math

import numpy as np
import pandas as pd
import pytest

from p1 import metrics

T0 = pd.Timestamp("2024-05-10 06:00")


def frame(x, y, index=None):
    index = (
        index if index is not None else pd.date_range(T0, periods=len(x), freq="min")
    )
    return pd.DataFrame({"X": x, "Y": y}, index=index, dtype=float)


def test_hand_example():
    truth = frame([0, 3, 0, 0], [0, 4, 0, 0])  # B_H = 0, 5, 0, 0
    pred = frame([1, 0, 3, 0], [0, 0, 4, 0])  # B_H = 1, 0, 5, 0
    m = metrics.score(pred, truth)
    # squared horizontal errors 1, 25, 25, 0
    assert m["n_scored"] == 4
    assert m["rmse_h"] == pytest.approx(math.sqrt(51 / 4))
    assert m["rmse_x"] == pytest.approx(math.sqrt(19 / 4))
    assert m["rmse_y"] == pytest.approx(math.sqrt(32 / 4))
    # both peaks are 5 nT; predicted one minute late
    assert m["peak_err"] == pytest.approx(0.0)
    assert m["peak_dt_min"] == 1.0
    # dB_H/dt: truth 5, 5, 0; prediction 1, 5, 5
    assert m["n_dbdt"] == 3
    assert m["dbdt_rmse"] == pytest.approx(math.sqrt((16 + 0 + 25) / 3))


def test_missing_minutes_are_not_scored():
    truth = frame([0, 6, 0, 0], [0, 8, 0, 0])  # peak 10 nT at minute 1
    pred = frame([0, 3, np.nan, 0], [0, 4, 1, 0])  # minute 2 has no X
    m = metrics.score(pred, truth)
    assert m["n_scored"] == 3
    assert m["rmse_h"] == pytest.approx(math.sqrt(25 / 3))
    assert m["peak_err"] == pytest.approx(5.0) and m["peak_dt_min"] == 0.0
    # only the step 0 -> 1 has both minutes scored: prediction 5, truth 10
    assert m["n_dbdt"] == 1 and m["dbdt_rmse"] == pytest.approx(5.0)


def test_dbdt_needs_consecutive_minutes():
    idx = pd.DatetimeIndex(
        [T0, T0 + pd.Timedelta(minutes=1), T0 + pd.Timedelta(minutes=3)]
    )
    truth = frame([0, 3, 6], [0, 4, 8], idx)
    pred = frame([0, 0, 0], [0, 0, 0], idx)
    m = metrics.score(pred, truth)
    assert m["n_dbdt"] == 1 and m["dbdt_rmse"] == pytest.approx(5.0)


def test_mask_and_common_mask():
    truth = frame([0, 3, np.nan, 0], [0, 4, 0, 0])
    a = frame([1, 0, 3, 0], [0, 0, 4, 0])
    b = frame([np.nan, 0, 0, 0], [0, 0, 0, 0])
    mask = metrics.common_mask(truth, a, b)
    assert mask.tolist() == [False, True, False, True]
    m = metrics.score(a, truth, mask=mask)
    assert m["n_scored"] == 2 and m["rmse_h"] == pytest.approx(math.sqrt(25 / 2))


def test_nothing_to_score():
    truth = frame([1, 2], [1, 2])
    pred = frame([np.nan, np.nan], [np.nan, np.nan])
    m = metrics.score(pred, truth)
    assert m["n_scored"] == 0 and m["n_dbdt"] == 0
    assert all(
        math.isnan(m[k])
        for k in ["rmse_h", "rmse_x", "rmse_y", "peak_err", "peak_dt_min", "dbdt_rmse"]
    )
