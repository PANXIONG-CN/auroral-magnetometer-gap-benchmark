"""Geometry, references, IDW, ridge, SECS and the evaluation runner on constructed cases."""

import math

import numpy as np
import pandas as pd
import pytest

from p1 import evaluate, geo, reference, secs
from p1.baselines import IDW, RidgeMap
from p1.scenarios import make_scenarios, split

T0 = pd.Timestamp("2017-08-01")


def frame(stations, minutes, fill=None, seed=0):
    idx = pd.date_range(T0, periods=minutes, freq="min", name="time")
    cols = pd.MultiIndex.from_product(
        [stations, ["X", "Y", "Z"]], names=["station", "component"]
    )
    rng = np.random.default_rng(seed)
    values = (
        rng.normal(size=(minutes, len(cols)))
        if fill is None
        else np.full((minutes, len(cols)), fill)
    )
    return pd.DataFrame(values, index=idx, columns=cols)


EQUATOR = pd.DataFrame(
    {"lat": [0.0, 0.0, 0.0], "lon": [0.0, 1.0, 2.0]}, index=["T", "A", "B"]
)


def test_geometry():
    one_degree = geo.R_EARTH_KM * math.pi / 180
    assert geo.distance_km(0, 0, 0, 1) == pytest.approx(one_degree)
    assert geo.distance_km(67.37, 26.63, 67.37, 26.63) == 0
    assert geo.bearing(0, 0, 1, 0) == pytest.approx(0.0)  # north
    assert geo.bearing(0, 0, 0, 1) == pytest.approx(math.pi / 2)  # east
    assert geo.bearing(0, 0, -1, 0) == pytest.approx(math.pi)


def test_references():
    data = frame(["A"], 3 * 1440, fill=0.0)
    data.loc["2017-08-02", ("A", "X")] = 10.0  # the quiet day; days 1 and 3 stay at 0
    assert reference.median_reference(data)[("A", "X")] == 0.0
    ref = reference.quiet_day_reference(data, [pd.Timestamp("2017-08-02")])
    assert ref[("A", "X")] == 10.0
    dist = reference.disturbance(data, ref)
    assert dist.loc["2017-08-03 00:00", ("A", "X")] == -10.0
    with pytest.raises(ValueError):
        reference.quiet_day_reference(data, [])


def scenario_inputs(data, stations, target, minutes):
    sc = make_scenarios(
        "E",
        T0,
        T0 + pd.Timedelta(minutes=minutes),
        stations,
        targets=[target],
        length_min=minutes,
    )[0]
    return split(sc, data, history_min=0)


def test_idw_hand_example():
    data = frame(["T", "A", "B"], 3, fill=0.0)
    data[("A", "X")] = [30.0, 30.0, np.nan]
    data[("B", "X")] = [0.0, 60.0, 60.0]
    inputs, _ = scenario_inputs(data, ["T", "A", "B"], "T", 3)
    pred = IDW(EQUATOR, power=1).predict(inputs)
    # A is half as far as B: weights 2:1
    assert pred["X"].tolist() == pytest.approx([20.0, 40.0, 60.0])


def test_ridge_recovers_linear_map_and_needs_all_inputs():
    data = frame(["T", "A", "B"], 2 * 1440, seed=1)
    data[("T", "X")] = 2 * data[("A", "X")] - data[("B", "Y")] + 5
    data[("T", "Y")] = 0.5 * data[("B", "X")]
    model = RidgeMap(alpha=1e-6).fit(data, "T", ["A", "B"])
    test = frame(["T", "A", "B"], 3, seed=2)
    test.iloc[2, test.columns.get_loc(("B", "Y"))] = np.nan
    inputs, _ = scenario_inputs(test, ["T", "A", "B"], "T", 3)
    pred = model.predict(inputs)
    expect_x = 2 * test[("A", "X")] - test[("B", "Y")] + 5
    assert pred["X"].iloc[:2].tolist() == pytest.approx(
        expect_x.iloc[:2].tolist(), abs=1e-6
    )
    assert pred.iloc[2].isna().all()
    cv = RidgeMap.day_cv(data, "T", ["A", "B"], [1e-6, 1e5])
    assert cv["rmse_h"].iloc[0] < 1e-3 < cv["rmse_h"].iloc[1]


def test_secs_field_matches_flat_sheet_limit():
    """Below a current sheet at height h, a DF SECS gives B_rho = mu0/(4 pi rho) (1 - h/sqrt(rho^2+h^2))
    in flat geometry. On the sphere the ratio tends to 1 + 2h/R_I for rho << R_E."""
    r, h = geo.R_EARTH_KM, 110.0
    e = h / (r + h)
    for rho in [5.0, 20.0, 50.0, 110.0, 200.0]:
        flat = (
            -(secs.MU0 / (4 * math.pi * rho * 1e3)) * (1 - h / math.hypot(rho, h)) * 1e9
        )
        ratio = secs.b_theta(rho / r, r, r + h) / flat
        assert ratio == pytest.approx(1 + 2 * e, abs=0.005)
    assert secs.b_theta(0.0, r, r + h) == 0.0 and secs.b_theta(math.pi, r, r + h) == 0.0


def test_secs_horizontal_field_points_along_the_great_circle():
    gx, gy = secs.design([67.0, 67.0], [20.0, 22.0], [68.0], [20.0], 110.0)
    assert abs(gy[0, 0]) < 1e-12 * abs(
        gx[0, 0]
    )  # pole due north: field purely north-south
    assert gx[0, 0] > 0  # with the sign of b_theta, the field points towards the pole


def test_secs_interpolates_observed_stations():
    coords = pd.DataFrame(
        {"lat": [66.0, 67.0, 68.5, 67.5], "lon": [20.0, 24.0, 22.0, 18.0]},
        index=["A", "B", "C", "D"],
    )
    poles = secs.pole_grid((63.0, 71.0, 0.5), (14.0, 30.0, 1.0))
    model = secs.SECS(coords, poles, epsilon=1e-10)
    op = model.operator("A", ("A", "B", "C", "D"))
    obs = np.array([10.0, -20.0, 5.0, 3.0, 7.0, 0.0, -4.0, 12.0])  # X of A..D, then Y
    assert op @ obs == pytest.approx([10.0, 7.0], abs=1e-6)


def test_secs_uses_the_valid_neighbours_of_each_minute():
    coords = pd.DataFrame(
        {"lat": [67.0, 66.0, 68.0, 67.5], "lon": [22.0, 20.0, 24.0, 18.0]},
        index=["T", "A", "B", "C"],
    )
    poles = secs.pole_grid((63.0, 71.0, 1.0), (14.0, 30.0, 2.0))
    model = secs.SECS(coords, poles, epsilon=0.01)
    data = frame(["T", "A", "B", "C"], 4, seed=3)
    data.iloc[1:3, data.columns.get_loc(("B", "X"))] = np.nan
    inputs, _ = scenario_inputs(data, ["T", "A", "B", "C"], "T", 4)
    pred = model.predict(inputs)
    full = model.operator("T", ("A", "B", "C"))
    part = model.operator("T", ("A", "C"))
    row0 = data.iloc[0]
    obs0 = [row0[(s, c)] for c in "XY" for s in "ABC"]
    row1 = data.iloc[1]
    obs1 = [row1[(s, c)] for c in "XY" for s in "AC"]
    assert pred.iloc[0].tolist() == pytest.approx(full @ obs0)
    assert pred.iloc[1].tolist() == pytest.approx(part @ obs1)


def test_runner_scores_a_perfect_prediction():
    data = frame(["T", "A", "B"], 360, seed=4)
    for s in ["A", "B"]:  # neighbours equal to the target: IDW reproduces it exactly
        data[(s, "X")] = data[("T", "X")]
        data[(s, "Y")] = data[("T", "Y")]
    scen = make_scenarios(
        "E", T0, T0 + pd.Timedelta(minutes=360), ["T", "A", "B"], targets=["T"]
    )
    runs = evaluate.run_method(IDW(EQUATOR), scen, data, coords=EQUATOR)
    rows = evaluate.score_runs(runs, method="idw", reference="none", station_set="toy")
    assert len(rows) == 2 and rows["rmse_h"].max() == pytest.approx(0.0, abs=1e-9)
    assert rows["n_inputs"].tolist() == [2.0, 2.0]
    assert rows["nn_distance_km"].iloc[0] == pytest.approx(
        geo.R_EARTH_KM * math.pi / 180
    )
    assert list(rows.columns[:6]) == [
        "event",
        "target",
        "scenario_start",
        "method",
        "reference",
        "station_set",
    ]
    masks = evaluate.common_masks(runs, runs)
    assert all(m.all() for m in masks)


def test_loo_select_ranks_parameters():
    rng = np.random.default_rng(5)
    data = frame(["T", "A", "B"], 360, fill=0.0)
    trend = rng.normal(size=360).cumsum()
    for s, scale in [("T", 1.0), ("A", 1.1), ("B", 3.0)]:  # B is the outlier station
        data[(s, "X")] = scale * trend
        data[(s, "Y")] = -scale * trend
    table = evaluate.loo_select(
        lambda p: IDW(EQUATOR, power=p),
        [0.5, 4.0],
        ["T", "A", "B"],
        data,
        T0,
        T0 + pd.Timedelta(minutes=360),
        coords=EQUATOR,
    )
    assert (
        table["param"].tolist() == [0.5, 4.0] and (table["n_scored"] == 3 * 360).all()
    )


def test_daily_medians_drop_poorly_covered_days():
    data = frame(["A", "B"], 2 * 1440, fill=1.0)
    data.loc["2017-08-02", ("A", "X")] = 5.0
    # 181 minutes of B missing on day 2: 87.4 % coverage
    data.loc["2017-08-02 00:00":"2017-08-02 03:00", ("B", "Y")] = np.nan
    med = reference.daily_medians(data, min_coverage=0.85)
    assert med.loc["2017-08-02", ("A", "X")] == 5.0
    assert med.loc["2017-08-01", ("B", "X")] == 1.0
    assert med.loc["2017-08-02", "B"].notna().all()
    med = reference.daily_medians(data, min_coverage=0.9)
    assert med.loc["2017-08-02", "B"].isna().all()
    assert med.loc["2017-08-02", "A"].notna().all()


def test_best_step_finds_a_step_and_not_noise():
    rng = np.random.default_rng(7)
    noise = rng.normal(scale=2.0, size=60)
    s = reference.best_step(noise + np.r_[np.zeros(25), np.full(35, 100.0)])
    assert s["k"] == 25 and s["step"] == pytest.approx(100.0, abs=2.0) and s["z"] > 50
    flat = reference.best_step(noise)
    assert flat["z"] < 5 and flat["explained"] < 0.3


def test_quiet_variation_recovers_the_daily_curve():
    data = frame(["A"], 3 * 1440, fill=0.0)
    minute = np.arange(3 * 1440) % 1440
    curve = 20 * np.sin(2 * np.pi * minute / 1440)
    data[("A", "X")] = curve + np.repeat(
        [100.0, 130.0, 500.0], 1440
    )  # day 3 is not quiet
    days = [pd.Timestamp("2017-08-01"), pd.Timestamp("2017-08-02")]
    var = reference.quiet_variation(data, days, smooth_min=1)
    assert var[("A", "X")].to_numpy() == pytest.approx(curve[:1440], abs=1e-9)
    const = reference.quiet_day_reference(data, days)
    ref = reference.reference_series(const, var, data.index[:3])
    assert ref[("A", "X")].tolist() == pytest.approx(const[("A", "X")] + curve[:3])


def test_nearest_neighbour_takes_the_closest_valid_station():
    from p1.baselines import NearestNeighbour

    data = frame(["T", "A", "B"], 3, fill=0.0)
    data[("A", "X")] = [1.0, np.nan, np.nan]
    data[("B", "X")] = [2.0, 2.0, np.nan]
    data.loc[data.index[2], ("B", "Y")] = np.nan
    inputs, _ = scenario_inputs(data, ["T", "A", "B"], "T", 3)
    pred = NearestNeighbour(EQUATOR).predict(inputs)
    assert pred["X"].iloc[0] == 1.0  # A is nearer
    assert pred["X"].iloc[1] == 2.0  # A missing: B
    assert pred.iloc[2].isna().all()  # nobody valid


def test_dynamic_ridge_uses_the_valid_inputs_of_each_minute():
    data = frame(["T", "A", "B"], 2 * 1440, seed=1)
    data[("T", "X")] = 2 * data[("A", "X")] + 5
    data[("T", "Y")] = 0.5 * data[("B", "Y")]
    test = frame(["T", "A", "B"], 3, seed=2)
    test.iloc[1, test.columns.get_loc(("B", "X"))] = np.nan  # B invalid in minute 1
    test.iloc[2, test.columns.get_loc(("A", "Y"))] = np.nan  # A invalid in minute 2
    inputs, _ = scenario_inputs(test, ["T", "A", "B"], "T", 3)
    fixed = RidgeMap(alpha=1e-6).fit(data, "T", ["A", "B"]).predict(inputs)
    assert fixed.iloc[1:].isna().all().all()
    dyn_model = RidgeMap(alpha=1e-6, dynamic=True).fit(data, "T", ["A", "B"])
    dyn = dyn_model.predict(inputs)
    assert dyn.iloc[0].tolist() == pytest.approx(fixed.iloc[0].tolist())
    assert dyn["X"].iloc[1] == pytest.approx(2 * test[("A", "X")].iloc[1] + 5, abs=1e-4)
    assert dyn.notna().all().all()
    assert set(dyn_model.models) == {("T", ("A", "B")), ("T", ("A",)), ("T", ("B",))}


def test_dynamic_ridge_drops_stations_without_training_data():
    """A station with no training minutes cannot enter a map; the ridge keeps predicting from
    the others instead of failing."""
    stations = ["T", "A", "B"]
    data = frame(stations, 600, seed=7)
    data[("T", "X")] = 2 * data[("A", "X")] + 0.5 * data[("B", "X")]
    data[("T", "Y")] = -data[("A", "Y")]
    train = data.copy()
    train[("B", "X")] = np.nan  # B never valid in training
    m = RidgeMap(alpha=1e-6, dynamic=True).fit(train, "T", ["A", "B"])
    inputs, _ = scenario_inputs(data, stations, "T", 180)
    pred = m.predict(inputs)
    assert pred.notna().all().all()
    assert ("T", ("A",)) in m.models and ("T", ("A", "B")) not in m.models
