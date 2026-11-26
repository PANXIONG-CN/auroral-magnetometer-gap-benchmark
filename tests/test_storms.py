"""Tests for storm verification and screening rules (hand-built series)."""

import numpy as np
import pandas as pd
import pytest

from p1 import storms


def hourly(start, values):
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq="h"), dtype=float
    )


def test_main_phase_and_window():
    # quiet 0 nT, SSC bump to +40 at 2024-05-10 17:00, decline to -400 at 2024-05-11 02:00, recovery
    v = np.zeros(24 * 4)
    v[24 + 17] = 40.0
    v[24 + 18 : 24 + 27] = np.linspace(-50, -400, 9)
    v[24 + 27 :] = -100.0
    dst = hourly("2024-05-09", v)
    mp = storms.main_phase(dst, "2024-05-10")
    assert mp["dst_min"] == -400 and mp["t_dst_min"] == pd.Timestamp("2024-05-11 02:00")
    assert mp["t_onset"] == pd.Timestamp("2024-05-10 17:00") and mp["main_phase_h"] == 9
    start, end = storms.window_from_onset(mp["t_onset"], 5)
    assert (start, end) == (pd.Timestamp("2024-05-10"), pd.Timestamp("2024-05-15"))


def kp_frame(start, values, definitive=True):
    idx = pd.date_range(start, periods=len(values), freq="3h")
    return pd.DataFrame({"kp": values, "definitive": definitive}, index=idx)


def test_quiet_days_and_kp_summary():
    day_quiet = [1.0, 2.333, 0.667, 1.333, 2.0, 1.0, 0.333, 2.333]  # max 2+
    day_unsettled = [1.0, 2.667, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # one 3-
    day_gap = [1.0] * 7 + [np.nan]
    day_storm = [3.0, 4.667, 5.0, 6.333, 7.0, 5.333, 4.0, 3.0]
    kp3h = kp_frame("2024-05-06", day_quiet + day_unsettled + day_gap + day_storm)
    q = storms.quiet_days(kp3h, "2024-05-09", 10, kp_max="2+")
    assert q == [pd.Timestamp("2024-05-06")]
    assert storms.quiet_days(kp3h, "2024-05-09", 10, kp_max="3-") == [
        pd.Timestamp("2024-05-06"),
        pd.Timestamp("2024-05-07"),
    ]
    s = storms.kp_summary(kp3h, "2024-05-09", "2024-05-10", active_kp=5)
    assert s["kp_max"] == 7.0 and s["t_kp_max"] == pd.Timestamp("2024-05-09 12:00")
    # 5o, 6+, 7o and 5+ count as active; 5- does not
    assert s["hours_kp_active"] == 12 and s["kp_definitive"]


def test_coverage_needs_both_components():
    idx = pd.date_range("2017-08-01", periods=10, freq="min")
    cols = pd.MultiIndex.from_product([["KEV", "SOD"], ["X", "Y", "Z"]])
    valid = pd.DataFrame(True, index=idx, columns=cols)
    valid.loc[idx[:3], ("SOD", "Y")] = False
    valid.loc[idx[:5], ("KEV", "Z")] = False  # Z does not matter
    cov = storms.coverage(
        valid, idx[0], idx[0] + pd.Timedelta(minutes=10), ["KEV", "SOD", "PEL"]
    )
    assert cov.to_dict() == {"KEV": 1.0, "SOD": pytest.approx(0.7), "PEL": 0.0}
    counts = storms.valid_station_counts(
        valid, idx[0], idx[0] + pd.Timedelta(minutes=10), ["KEV", "SOD"]
    )
    assert list(counts) == [1, 1, 1] + [2] * 7


def mask(start, minutes, stations):
    idx = pd.date_range(start, periods=minutes, freq="min")
    cols = pd.MultiIndex.from_product([stations, ["X", "Y", "Z"]])
    return pd.DataFrame(True, index=idx, columns=cols)


def test_quiet_day_coverage_takes_best_day():
    v = mask("2024-05-01", 2 * 1440, ["KEV", "SOD"])
    v.loc["2024-05-01 00:00":"2024-05-01 11:59", ("SOD", "X")] = False  # day 1: 50 %
    v.loc["2024-05-02 00:00":"2024-05-02 02:23", ("SOD", "Y")] = False  # day 2: 90 %
    days = [pd.Timestamp("2024-05-01"), pd.Timestamp("2024-05-02")]
    cov = storms.quiet_day_coverage(v, days, ["KEV", "SOD"])
    assert cov["KEV"] == 1.0 and cov["SOD"] == pytest.approx(0.9)
    assert storms.quiet_day_coverage(v, [], ["KEV"]).isna().all()


def test_per_minute_rule():
    st = ["A", "B", "C"]
    v = mask("2024-05-10", 100, st)
    v.loc[v.index[:2], ("A", "X")] = False  # 2 of 100 minutes with only B, C valid
    end = v.index[0] + pd.Timedelta(minutes=100)
    r = storms.per_minute_rule(v, v.index[0], end, st, min_valid=3, min_frac=0.98)
    assert r == {"min_valid_per_minute": 2, "frac_minutes_ok": 0.98, "ok": True}
    r = storms.per_minute_rule(v, v.index[0], end, st, min_valid=3, min_frac=0.99)
    assert not r["ok"]
    r = storms.per_minute_rule(v, v.index[0], end, [], min_valid=1, min_frac=0.5)
    assert r == {"min_valid_per_minute": 0, "frac_minutes_ok": 0.0, "ok": False}
