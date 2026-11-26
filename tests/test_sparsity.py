"""Sparsification helpers: convex hull, binning, relative error and the two-segment fit."""

import numpy as np
import pandas as pd

from p1 import sparsity

COORDS = pd.DataFrame(
    {"lat": [67.0, 66.0, 66.0, 68.0, 70.0], "lon": [25.0, 23.0, 27.0, 25.0, 25.0]},
    index=["T", "A", "B", "C", "D"],
)


def test_inside_hull():
    assert sparsity.inside_hull(COORDS, "T", ["A", "B", "C"])
    assert not sparsity.inside_hull(
        COORDS, "T", ["A", "C", "D"]
    )  # collinear-ish west side
    assert not sparsity.inside_hull(COORDS, "D", ["A", "B", "C"])
    assert not sparsity.inside_hull(COORDS, "T", ["A", "B"])


def test_hinge_fit_recovers_the_break():
    x = np.arange(1.0, 9.0)
    y = np.where(x < 4, 1 + 0.1 * x, 1.4 + 0.6 * (x - 4))
    fit = sparsity.hinge_fit(x, y, np.ones_like(x), [3.0, 4.0, 5.0, 6.0])
    assert fit["hinge"] == 4.0
    assert np.isclose(fit["before"], 0.1) and np.isclose(fit["after"], 0.6)


def test_relative_error_pairs_scenarios_with_the_full_network():
    rows = pd.DataFrame(
        {
            "event": ["e"] * 4,
            "target": ["T"] * 4,
            "scenario_start": [0, 1, 0, 1],
            "method": ["secs"] * 4,
            "k": [0, 0, 2, 2],
            "config": ["near0", "near0", "near2", "near2"],
            "rmse_h": [10.0, 20.0, 20.0, 20.0],
            "n_scored": [100, 100, 100, 100],
            "nn_bin": ["a", "a", "b", "b"],
        }
    )
    rel = sparsity.relative_error(rows).set_index("nn_bin")["relative"]
    assert "a" not in rel.index  # the full network is the denominator only
    assert np.isclose(rel["b"], np.sqrt((400 + 400) / (100 + 400)))
    assert sparsity.relative_error(rows, exclude=["T"]).empty


def test_configurations_are_deterministic_and_drop_the_nearest_first():
    from types import SimpleNamespace

    from p1 import config, geo, pipeline

    files = config.station_set("leave_one_out_12")
    s = SimpleNamespace(
        files=files,
        coords=geo.station_coords(files),
        cfg={
            "sparsify": {
                "max_removed": 8,
                "min_inputs": 3,
                "random_sizes": [3, 5],
                "random_draws": 4,
                "seed": 0,
            }
        },
    )
    a = pipeline.configurations(s, "SOD")
    b = pipeline.configurations(s, "SOD")
    assert a == b
    near = [c for c in a if c["family"] == "nearest_removed"]
    assert [c["k"] for c in near] == list(range(9))
    assert (
        "PEL" in near[0]["inputs"] and "PEL" not in near[1]["inputs"]
    )  # PEL is nearest to SOD
    assert near[1]["nominal_nn_km"] > near[0]["nominal_nn_km"]
    rand = [c for c in a if c["family"] == "random"]
    assert len(rand) == 8 and len({c["inputs"] for c in rand}) == 8
    assert all("SOD" not in c["inputs"] for c in a)
