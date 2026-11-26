"""Quiet constant (D14 coverage), whole-window runs, temporal methods and station configurations."""

import numpy as np
import pandas as pd
import pytest

from p1 import evaluate, reference, secs
from p1.baselines import IDW
from p1.scenarios import make_scenarios, split
from p1.temporal import KalmanSECS, ResidualCorrection

T0 = pd.Timestamp("2017-08-01")
COORDS = pd.DataFrame(
    {
        "lat": [67.4, 69.3, 69.7, 69.8, 68.4, 66.9],
        "lon": [26.6, 16.0, 18.9, 27.0, 18.8, 24.1],
    },
    index=["T", "A", "B", "C", "D", "E"],
)
POLES = secs.pole_grid((64.0, 72.0, 1.0), (12.0, 32.0, 2.0))


def frame(stations, minutes, *, seed=0, start=T0):
    idx = pd.date_range(start, periods=minutes, freq="min", name="time")
    cols = pd.MultiIndex.from_product(
        [stations, ["X", "Y", "Z"]], names=["station", "component"]
    )
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(minutes, len(cols))), index=idx, columns=cols)


def secs_field(stations, minutes, *, seed=0, start=T0, height_km=110.0):
    """Ground X, Y from random SECS amplitudes that drift slowly in time."""
    gx, gy = secs.design(
        COORDS.loc[stations, "lat"],
        COORDS.loc[stations, "lon"],
        POLES.lat,
        POLES.lon,
        height_km,
    )
    rng = np.random.default_rng(seed)
    amp = np.cumsum(rng.normal(size=(minutes, len(POLES))), axis=0) * 2e3
    data = frame(stations, minutes, seed=seed, start=start)
    for i, s in enumerate(stations):
        data[(s, "X")] = amp @ gx[i]
        data[(s, "Y")] = amp @ gy[i]
    return data


def test_quiet_constant_uses_only_well_covered_days():
    data = frame(["A"], 2 * 1440)
    data.loc["2017-08-01", ("A", "X")] = 10.0
    data.loc["2017-08-02", ("A", "X")] = 50.0
    data.loc["2017-08-02 00:00":"2017-08-02 11:59", ("A", "Y")] = (
        np.nan
    )  # 50 % coverage
    days = [pd.Timestamp("2017-08-01"), pd.Timestamp("2017-08-02")]
    assert reference.quiet_constant(data, days)[("A", "X")] == 10.0
    assert reference.quiet_constant(data, days, min_coverage=0.4)[("A", "X")] == 30.0
    data.loc["2017-08-01 00:00":"2017-08-01 23:00", ("A", "Y")] = np.nan
    assert np.isnan(reference.quiet_constant(data, days)[("A", "X")])


def test_decomposed_reference_adds_core_change_and_daily_curve():
    data = frame(["A"], 3 * 1440) * 0.0
    mod = np.arange(3 * 1440) % 1440
    data[("A", "X")] = 100.0 + 5.0 * np.sin(2 * np.pi * mod / 1440)
    days = [T0, T0 + pd.Timedelta(days=1)]
    idx = pd.date_range(T0 + pd.Timedelta(days=2), periods=1440, freq="min")
    core = pd.Series(
        [2.0, 0.0, 0.0], index=pd.MultiIndex.from_product([["A"], ["X", "Y", "Z"]])
    )
    ref = reference.decomposed_reference(
        data, days, idx, core, smooth_min=1, min_variation_days=1
    )
    expect = 100.0 + 2.0 + 5.0 * np.sin(2 * np.pi * np.arange(1440) / 1440)
    assert np.allclose(ref[("A", "X")].to_numpy(), expect, atol=1e-6)


@pytest.mark.parametrize("method", ["idw", "secs"])
def test_run_window_equals_scenario_runs(method):
    st = ["T", "A", "B", "C", "D", "E"]
    data = frame(st, 360, seed=1)
    data.iloc[30:200, data.columns.get_loc(("B", "X"))] = np.nan
    model = (
        IDW(COORDS, power=2) if method == "idw" else secs.SECS(COORDS, POLES, 110, 0.05)
    )
    end = T0 + pd.Timedelta(minutes=360)
    scen = make_scenarios("E", T0, end, st, targets=["T"])
    a = evaluate.run_method(model, scen, data, coords=COORDS)
    b = evaluate.run_window(model, "E", T0, end, "T", st[1:], data, coords=COORDS)
    assert len(a) == len(b) == 2
    for ra, rb in zip(a, b, strict=True):
        key = ("event", "target", "start", "end", "neighbours")
        assert [getattr(ra["scenario"], k) for k in key] == [
            getattr(rb["scenario"], k) for k in key
        ]  # kp is NaN here, and NaN != NaN
        pd.testing.assert_frame_equal(ra["pred"], rb["pred"], check_freq=False)
        pd.testing.assert_frame_equal(ra["truth"], rb["truth"], check_freq=False)
        assert np.array_equal(ra["n_inputs"], rb["n_inputs"])


def gap_inputs(data, neighbours, history_min, start_min=240, length=180):
    start = T0 + pd.Timedelta(minutes=start_min)
    sc = make_scenarios(
        "E",
        start,
        start + pd.Timedelta(minutes=length),
        ["T", *neighbours],
        targets=["T"],
    )[0]
    return split(sc, data, history_min=history_min)


def test_kalman_static_limit_is_tikhonov_and_ignores_history():
    neigh = ["A", "B", "C", "D", "E"]
    data = secs_field(["T", *neigh], 420, seed=2)
    inputs, _ = gap_inputs(data, neigh, history_min=60)
    lam = 0.01
    kf = KalmanSECS(COORDS, POLES, 110, a=0.0, lam=lam)
    pred = kf.predict(inputs)
    gx, gy = secs.design(COORDS.lat, COORDS.lon, POLES.lat, POLES.lon, 110)
    rows = [COORDS.index.get_loc(s) for s in neigh]
    g_all = np.vstack([gx[[*rows, 0]], gy[[*rows, 0]]])
    smax = np.linalg.svd(g_all, compute_uv=False)[0]
    gn = np.vstack([gx[rows], gy[rows]])
    gt = np.vstack([gx[0], gy[0]])
    op = gt @ gn.T @ np.linalg.inv(gn @ gn.T + lam * smax**2 * np.eye(len(gn)))
    x = inputs.current.xs("X", axis=1, level="component")[neigh].to_numpy()
    y = inputs.current.xs("Y", axis=1, level="component")[neigh].to_numpy()
    expect = np.hstack([x, y]) @ op.T
    assert np.allclose(pred.to_numpy(), expect, rtol=1e-6, atol=1e-6)
    # with a = 0 the history does not matter
    inputs2, _ = gap_inputs(data, neigh, history_min=0)
    assert np.allclose(kf.predict(inputs2).to_numpy(), pred.to_numpy())


def test_kalman_memory_carries_target_information_into_the_gap():
    """A field that the neighbours barely see at T: with memory the filter keeps what it learnt
    from T before the gap, so its error is below the static limit at the start of the gap."""
    neigh = ["A", "B", "D"]
    data = secs_field(["T", *neigh], 420, seed=3)
    inputs, truth = gap_inputs(data, neigh, history_min=240)
    tv = truth.values[["X", "Y"]].to_numpy()
    static = KalmanSECS(COORDS, POLES, 110, a=0.0, lam=1e-3).predict(inputs).to_numpy()
    memory = (
        KalmanSECS(COORDS, POLES, 110, a=0.999, lam=1e-3).predict(inputs).to_numpy()
    )
    early = slice(0, 30)
    err_s = np.sqrt(np.mean((static[early] - tv[early]) ** 2))
    err_m = np.sqrt(np.mean((memory[early] - tv[early]) ** 2))
    assert err_m < 0.5 * err_s


def test_kalman_grid_matches_single_runs():
    neigh = ["A", "B", "C"]
    data = secs_field(["T", *neigh], 300, seed=4)
    inputs, _ = gap_inputs(data, neigh, history_min=60, start_min=120)
    grid = KalmanSECS(COORDS, POLES, 110, a=[0.0, 0.95], lam=[0.01, 0.001]).run(inputs)
    one = KalmanSECS(COORDS, POLES, 110, a=0.95, lam=0.001).predict(inputs).to_numpy()
    assert np.allclose(grid[1], one)


def test_residual_correction_removes_a_constant_offset_using_history_only():
    neigh = ["A", "B", "C", "D", "E"]
    data = frame(["T", *neigh], 420, seed=5)
    base = IDW(COORDS, power=2)
    # the target is the IDW combination of its neighbours plus 7 nT in X and -3 nT in Y
    whole = make_scenarios(
        "E",
        T0,
        T0 + pd.Timedelta(minutes=420),
        ["T", *neigh],
        targets=["T"],
        length_min=420,
    )[0]
    inp, _ = split(whole, data, history_min=0)
    p = base.predict(inp)
    data[("T", "X")] = p["X"].to_numpy() + 7.0
    data[("T", "Y")] = p["Y"].to_numpy() - 3.0
    inputs, truth = gap_inputs(data, neigh, history_min=60)
    for stat in ["mean", "last"]:
        rc = ResidualCorrection(base, window_min=30, statistic=stat)
        assert np.allclose(rc.correction(inputs), [7.0, -3.0])
        assert np.allclose(
            rc.predict(inputs).to_numpy(), truth.values[["X", "Y"]].to_numpy()
        )
    # values inside the gap never reach the correction
    data.loc[T0 + pd.Timedelta(minutes=240) :, ("T", "X")] += 1000.0
    inputs, _ = gap_inputs(data, neigh, history_min=60)
    assert np.allclose(
        ResidualCorrection(base, window_min=30).correction(inputs), [7.0, -3.0]
    )


def test_decomposed_reference_uses_the_network_curve_for_stations_with_few_quiet_days():
    days = [T0 + pd.Timedelta(days=i) for i in range(3)]
    data = frame(["A", "B", "C"], 4 * 1440) * 0.0
    mod = np.arange(4 * 1440) % 1440
    curve = 5.0 * np.sin(2 * np.pi * mod / 1440)
    for st in ["A", "B"]:
        data[(st, "X")] = curve
    data[("C", "X")] = curve + np.where(mod == 600, -500.0, 0.0)  # a spike on every day
    data.loc[T0 : T0 + pd.Timedelta(hours=47, minutes=59), ("C", "Y")] = (
        np.nan
    )  # 1 good day
    idx = pd.date_range(T0 + pd.Timedelta(days=3), periods=1440, freq="min")
    core = pd.Series(0.0, index=data.columns)
    ref = reference.decomposed_reference(data, days, idx, core, smooth_min=1)
    expect = 5.0 * np.sin(2 * np.pi * np.arange(1440) / 1440)
    assert np.allclose(
        ref[("C", "X")].to_numpy() - ref[("C", "X")].median(),
        expect - np.median(expect),
        atol=1e-6,
    )
