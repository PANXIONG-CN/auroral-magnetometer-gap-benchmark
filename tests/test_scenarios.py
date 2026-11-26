"""Scenario generation and hidden-station isolation.

T2 acceptance: the hidden station's truth cannot be reached through the input interface.
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from p1 import scenarios as sc

ST = ["AND", "KEV", "SOD"]
MIN = pd.Timedelta(minutes=1)


def event_data(start="2024-05-10", days=1, history_days=1):
    idx = pd.date_range(
        pd.Timestamp(start) - pd.Timedelta(days=history_days),
        periods=(days + history_days) * 1440,
        freq="min",
        name="time",
    )
    cols = pd.MultiIndex.from_product(
        [ST, ["X", "Y", "Z"]], names=["station", "component"]
    )
    rng = np.random.default_rng(1)
    return pd.DataFrame(rng.normal(size=(len(idx), len(cols))), index=idx, columns=cols)


def test_make_scenarios_blocks_and_neighbours():
    s = sc.make_scenarios("E1", "2024-05-10", "2024-05-15", ST)
    assert len(s) == 3 * 40
    for target in ST:
        mine = [x for x in s if x.target == target]
        assert len(mine) == 40
        assert all(target not in x.neighbours and len(x.neighbours) == 2 for x in mine)
        assert mine[0].start == pd.Timestamp("2024-05-10")
        assert mine[-1].end == pd.Timestamp("2024-05-15")
        assert all(a.end == b.start for a, b in itertools.pairwise(mine))
        assert all(x.end - x.start == pd.Timedelta(minutes=180) for x in mine)


def test_make_scenarios_rejects_bad_input():
    with pytest.raises(ValueError):
        sc.make_scenarios("E1", "2024-05-10", "2024-05-10 04:00", ST)
    with pytest.raises(ValueError):
        sc.make_scenarios("E1", "2024-05-10", "2024-05-11", ST, targets=["PEL"])


def test_kp_per_scenario_and_group():
    kp3h = pd.DataFrame(
        {"kp": [4.667, 5.0, 2.0]},
        index=pd.date_range("2024-05-10", periods=3, freq="3h"),
    )
    s = sc.make_scenarios(
        "E1", "2024-05-10", "2024-05-10 09:00", ST, targets=["SOD"], kp3h=kp3h
    )
    assert [x.kp for x in s] == [4.667, 5.0, 2.0]
    assert [sc.kp_group(x.kp) for x in s] == ["Kp<5", "Kp>=5", "Kp<5"]
    assert sc.kp_group(float("nan")) == "unknown"


def scenario_at(hour=6):
    s = sc.make_scenarios("E1", "2024-05-10", "2024-05-11", ST, targets=["SOD"])
    return s[hour // 3]


def test_split_hides_the_target_inside_the_window():
    data = event_data()
    scen = scenario_at(6)
    sentinel = 12345.678
    data.loc[scen.start : scen.end - MIN, "SOD"] = sentinel
    inputs, truth = sc.split(scen, data, history_min=600)

    assert "SOD" not in inputs.current.columns.get_level_values("station")
    assert inputs.history.index.max() < scen.start
    assert set(vars(inputs)) == {"scenario", "current", "history"}
    for df in (inputs.current, inputs.history):
        assert not np.isclose(df.to_numpy(), sentinel).any()
    assert (
        truth.values.shape == (180, 3) and (truth.values.to_numpy() == sentinel).all()
    )


def test_split_history_keeps_the_target_before_the_gap():
    data = event_data()
    scen = scenario_at(6)
    inputs, _ = sc.split(scen, data, history_min=600)
    assert (
        inputs.history.index[0] == scen.start - 600 * MIN and len(inputs.history) == 600
    )
    expected = data.loc[scen.start - 600 * MIN : scen.start - MIN, "SOD"]
    np.testing.assert_array_equal(inputs.history["SOD"].to_numpy(), expected.to_numpy())
    np.testing.assert_array_equal(
        inputs.current["KEV"].to_numpy(),
        data.loc[scen.start : scen.end - MIN, "KEV"].to_numpy(),
    )


def test_split_returns_copies():
    data = event_data()
    before = data.copy()
    inputs, truth = sc.split(scenario_at(6), data, history_min=60)
    inputs.current.iloc[:, :] = 0.0
    inputs.history.iloc[:, :] = 0.0
    truth.values.iloc[:, :] = 0.0
    pd.testing.assert_frame_equal(data, before)
    _, truth2 = sc.split(scenario_at(6), data, history_min=60)
    assert not (truth2.values.to_numpy() == 0.0).all()


def test_split_covers_every_minute_and_empty_history():
    data = event_data()
    scen = scenario_at(6)
    gap = (data.index >= scen.start) & (data.index < scen.start + 10 * MIN)
    inputs, truth = sc.split(scen, data[~gap], history_min=0)
    assert len(inputs.current) == 180 and len(truth.values) == 180
    assert inputs.current.iloc[:10].isna().all().all()
    assert truth.values.iloc[:10].isna().all().all()
    assert inputs.history.empty
