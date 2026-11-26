"""Sparsification curves, the gate-A judgement and the C2 test (E009; reused by E011).

Scenario rows come from p1.pipeline.score_common with the configuration columns of
p1.pipeline.configurations. Each scenario falls in a nearest-neighbour distance bin by its median
nearest valid neighbour distance over the scored minutes. Event-level values pool scored minutes
(D15); intervals resample events (hard rule 5).

- relative SECS error of a bin: sqrt(sum n rmse^2 / sum n rmse0^2) over the bin's scenarios in
  reduced networks, rmse0 being the same target and scenario in the full network (near0);
- gate A: a continuous two-segment line through the event-mean relative error of the bins,
  hinge at a bin edge, weighted by scored minutes; a rapid rise needs the slope after the hinge
  to be at least twice the slope before and its event-bootstrap interval to lie above 0;
- C2: the slope of (other method - SECS) against the bin centre within each event, and the
  difference pooled over scenarios at or beyond the threshold d*.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from . import geo, stats


def inside_hull(coords: pd.DataFrame, target: str, inputs) -> bool:
    """Whether the target lies inside the convex hull of the input stations (local km plane)."""
    lat0 = np.radians(coords["lat"].mean())
    k = geo.R_EARTH_KM * np.pi / 180

    def xy(names):
        c = coords.loc[list(names)]
        return np.column_stack([c["lon"] * k * np.cos(lat0), c["lat"] * k])

    pts = xy(inputs)
    if len(pts) < 3:
        return False
    try:
        tri = Delaunay(pts)
    except RuntimeError:  # QhullError: collinear inputs
        return False
    return bool(tri.find_simplex(xy([target]))[0] >= 0)


def add_bins(rows: pd.DataFrame, edges) -> pd.DataFrame:
    edges = list(edges)
    labels = [f"{a}-{b}" for a, b in itertools.pairwise(edges)]
    out = rows.copy()
    out["nn_bin"] = pd.cut(
        out["nn_distance_km"], edges, labels=labels, right=False
    ).astype(str)
    out["n_inputs_int"] = out["n_inputs"].round().astype("Int64")
    return out


def bin_centres(rows: pd.DataFrame) -> pd.Series:
    """Minute-weighted mean nearest-neighbour distance per bin."""
    r = rows[rows["n_scored"] > 0]
    w = r["n_scored"].astype(float)
    return (r["nn_distance_km"] * w).groupby(r["nn_bin"]).sum() / w.groupby(
        r["nn_bin"]
    ).sum()


def relative_error(
    rows: pd.DataFrame, method: str = "secs", *, exclude=()
) -> pd.DataFrame:
    """Per (event, nn_bin): relative error of `method` against the full network (config near0).

    The full-network scenarios are the denominators and are left out of the numerators (in the
    first E009 run they were included, which pinned their bins towards 1). Targets in `exclude`
    are left out (E009: OUJ, outside the network, whose full-network error is already 99 nT).
    """
    r = rows[(rows["method"] == method) & (rows["n_scored"] > 0)]
    r = r[~r["target"].isin(list(exclude))]
    full = r[r["config"] == "near0"].set_index(["event", "target", "scenario_start"])[
        "rmse_h"
    ]
    r = r[r["config"] != "near0"]
    key = pd.MultiIndex.from_frame(r[["event", "target", "scenario_start"]])
    r = r.assign(rmse0=full.reindex(key).to_numpy())
    r = r[r["rmse0"].notna()]
    num = (r["n_scored"] * r["rmse_h"] ** 2).groupby([r["event"], r["nn_bin"]]).sum()
    den = (r["n_scored"] * r["rmse0"] ** 2).groupby([r["event"], r["nn_bin"]]).sum()
    n = r.groupby(["event", "nn_bin"])["n_scored"].sum()
    return pd.DataFrame({"relative": np.sqrt(num / den), "n_scored": n}).reset_index()


def _hinge_design(x, h):
    x = np.asarray(x, float)
    return np.column_stack([np.ones_like(x), x, np.maximum(x - h, 0.0)])


def hinge_fit(x, y, w, hinges) -> dict:
    """Weighted continuous two-segment fit; the hinge minimising the weighted SSE."""
    x, y, w = (np.asarray(v, float) for v in (x, y, w))
    best = None
    for h in hinges:
        if (x < h).sum() < 2 or (x >= h).sum() < 2:
            continue
        a = _hinge_design(x, h)
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(a * sw[:, None], y * sw, rcond=None)
        sse = float(np.sum(w * (a @ coef - y) ** 2))
        if best is None or sse < best["sse"]:
            best = {
                "hinge": float(h),
                "before": coef[1],
                "after": coef[1] + coef[2],
                "sse": sse,
            }
    return best


def gate_a(
    rel: pd.DataFrame, centres: pd.Series, edges, *, n_resamples=10000, seed=0
) -> dict:
    """Two-segment fit of the event-mean relative SECS error; slopes per 100 km, with event-
    bootstrap intervals at the chosen hinge."""
    wide = rel.pivot(index="nn_bin", columns="event", values="relative")
    nmin = rel.groupby("nn_bin")["n_scored"].sum()
    wide = wide.loc[[b for b in wide.index if b in centres.index]].dropna()
    x = centres.reindex(wide.index).to_numpy() / 100.0
    w = nmin.reindex(wide.index).to_numpy(float)
    hinges = [e / 100.0 for e in list(edges)[1:-1]]
    fit = hinge_fit(x, wide.mean(axis=1).to_numpy(), w, hinges)
    h = fit["hinge"]
    events = wide.columns.to_numpy()
    rng = np.random.default_rng(seed)
    a = _hinge_design(x, h)
    sw = np.sqrt(w)
    boot = []
    for _ in range(n_resamples):
        pick = rng.integers(0, len(events), len(events))
        y = wide[events[pick]].mean(axis=1).to_numpy()
        coef, *_ = np.linalg.lstsq(a * sw[:, None], y * sw, rcond=None)
        boot.append((coef[1], coef[1] + coef[2]))
    boot = np.array(boot)
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    rapid = bool(fit["after"] >= 2 * fit["before"] and lo[1] > 0)
    return {
        "hinge_km": h * 100,
        "slope_before_per_100km": fit["before"],
        "slope_before_lo": lo[0],
        "slope_before_hi": hi[0],
        "slope_after_per_100km": fit["after"],
        "slope_after_lo": lo[1],
        "slope_after_hi": hi[1],
        "rapid_rise": rapid,
        "curve": pd.DataFrame(
            {"centre_km": x * 100, "relative_mean": wide.mean(axis=1), "n_scored": w}
        ),
    }


def c2_tests(
    ev_bins: pd.DataFrame, centres: pd.Series, rows_far: pd.DataFrame, others
) -> pd.DataFrame:
    """Per other method: slope of (other - SECS) against the bin centre (nT per 100 km) within each
    event, and the difference pooled over the scenarios at or beyond d*."""
    out = []
    far = stats.event_summary(rows_far, by=["event", "method"])
    for other in others:
        slopes = []
        for _, g in ev_bins.groupby("event"):
            wide = g.pivot(index="nn_bin", columns="method", values="rmse_h")
            n = g[g["method"] == "secs"].set_index("nn_bin")["n_scored"]
            d = (wide[other] - wide["secs"]).dropna()
            x = centres.reindex(d.index).to_numpy() / 100.0
            w = n.reindex(d.index).to_numpy(float)
            if len(d) < 2:
                continue
            a = np.column_stack([np.ones_like(x), x]) * np.sqrt(w)[:, None]
            coef, *_ = np.linalg.lstsq(a, d.to_numpy() * np.sqrt(w), rcond=None)
            slopes.append(coef[1])
        b = stats.bootstrap_mean(slopes)
        d = stats.paired_difference(far, "rmse_h", other, "secs")
        out.append(
            {
                "comparison": f"{other} - secs",
                "slope_nT_per_100km": b["mean"],
                "slope_lo": b["lo"],
                "slope_hi": b["hi"],
                "sparse_end_nT": d["mean"],
                "sparse_end_lo": d["lo"],
                "sparse_end_hi": d["hi"],
                "per_event_slope": " ".join(f"{v:.2f}" for v in slopes),
            }
        )
    return pd.DataFrame(out)
