"""Verdict rules of the locked test (docs/01_research_design.md), on constructed tables."""

import pandas as pd

from p1 import final_eval


def test_verdict_c2():
    t = pd.DataFrame(
        [
            {
                "comparison": "a - secs",
                "slope_lo": 0.5,
                "slope_hi": 2,
                "sparse_end_lo": 1,
                "sparse_end_hi": 5,
            },
            {
                "comparison": "b - secs",
                "slope_lo": 0.5,
                "slope_hi": 2,
                "sparse_end_lo": -1,
                "sparse_end_hi": 5,
            },
            {
                "comparison": "c - secs",
                "slope_lo": -3,
                "slope_hi": -1,
                "sparse_end_lo": 1,
                "sparse_end_hi": 5,
            },
            {
                "comparison": "d - secs",
                "slope_lo": -1,
                "slope_hi": 1,
                "sparse_end_lo": 1,
                "sparse_end_hi": 5,
            },
        ]
    )
    assert final_eval.verdict_c2(t) == {
        "a - secs": "supported",
        "b - secs": "grows_sparse_end_not_significant",
        "c - secs": "gain_shrinks_with_sparsity",
        "d - secs": "gain_independent_of_density",
    }


def gains(dense, sparse, method="kalman"):
    return [
        {
            "kp_group": "all",
            "station_set": "dense",
            "comparison": f"secs - {method}",
            "lo": dense[0],
            "hi": dense[1],
        },
        {
            "kp_group": "all",
            "station_set": "sparse",
            "comparison": f"secs - {method}",
            "lo": sparse[0],
            "hi": sparse[1],
        },
    ]


def test_verdict_c3():
    cases = {
        ((-1, 1), (0.5, 2)): "supported",
        ((0.5, 2), (0.5, 2)): "gain_in_both",
        ((0.5, 2), (-1, 1)): "gain_only_dense",
        ((-3, -1), (-3, -1)): "worse_in_both",
        ((-1, 1), (-1, 1)): "no_gain_in_either",
        ((-3, -1), (-1, 1)): "no_gain_in_either",
    }
    for (d, sp), expect in cases.items():
        assert final_eval.verdict_c3(pd.DataFrame(gains(d, sp))) == {"kalman": expect}


def test_flag_stations_uses_the_network_relative_change():
    bias = pd.DataFrame(
        {
            "bias_change_X": [5.0, 6.0, 4.0, -20.0],
            "bias_change_Y": [50.0, 52.0, 49.0, 51.0],
        },
        index=["A", "B", "C", "D"],
    )
    # a common shift of 50 nT in Y is the main field, not a station problem
    assert final_eval.flag_stations(bias, 10) == ["D"]
    assert "relative_X" in bias.columns


def row(mean, lo, n_pos, n=9):
    return pd.Series({"mean": mean, "lo": lo, "n_positive": n_pos, "events": n})


def test_verdict_c1():
    v = final_eval.verdict_c1
    assert v(row(30, 10, 9), row(25, 8, 9), any_flagged=True)["verdict"] == "supported"
    assert (
        v(row(30, 10, 9), row(2, -1, 5), any_flagged=True)["verdict"]
        == "few_stations_driven"
    )
    assert (
        v(row(30, 10, 9), row(10, 2, 9), any_flagged=True)["verdict"]
        == "few_stations_driven"
    )
    assert (
        v(row(30, 10, 9), row(30, 10, 9), any_flagged=False)["verdict"] == "supported"
    )
    assert (
        v(row(30, -2, 7), row(30, -2, 7), any_flagged=False)["verdict"]
        == "not_all_events"
    )
    assert (
        v(row(30, 10, 9), row(10, 2, 9), any_flagged=True)["remaining_share"] == 10 / 30
    )
