"""Run reconstruction methods over scenarios and score them (protocol sections 3, 5 and 7).

run_method returns predictions with the hidden truth kept apart: `truth` is filled by split() and
used only by score_runs. Result rows carry the columns of protocol section 7 plus n_scored and
n_dbdt, which the event-level pooling in p1.stats needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import geo, metrics
from .baselines import components
from .scenarios import Scenario, kp_group, make_scenarios, split

RESULT_COLUMNS = [
    "event", "target", "scenario_start", "method", "reference", "station_set", "n_inputs",
    "nn_distance_km", "rmse_h", "rmse_x", "rmse_y", "peak_err", "peak_dt_min", "dbdt_rmse",
    "kp_group", "causal", "git", "cfg", "n_scored", "n_dbdt",
]  # fmt: skip


def run_method(
    method, scenarios, data, *, coords: pd.DataFrame, history_min: int = 0
) -> list[dict]:
    """Predict every scenario. Each item: scenario, pred, truth (X, Y), n_inputs and nn_km per minute."""
    runs = []
    for sc in scenarios:
        inputs, truth = split(sc, data, history_min=history_min)
        pred = method.predict(inputs)
        x, y = components(inputs.current, sc.neighbours)
        ok = ~(np.isnan(x) | np.isnan(y))
        t = coords.loc[sc.target]
        neigh = coords.loc[list(sc.neighbours)]
        d = np.asarray(geo.distance_km(t.lat, t.lon, neigh.lat, neigh.lon), float)
        nn = np.where(ok, d[None, :], np.inf).min(axis=1)
        nn[~ok.any(axis=1)] = np.nan
        runs.append(
            {
                "scenario": sc,
                "pred": pred,
                "truth": truth.values[["X", "Y"]],
                "n_inputs": ok.sum(axis=1),
                "nn_km": nn,
            }
        )
    return runs


def run_window(
    method,
    event,
    start,
    end,
    target,
    neighbours,
    data,
    *,
    coords,
    kp3h=None,
    length_min: int = 180,
) -> list[dict]:
    """run_method for a spatial method, predicting the whole window [start, end) in one call.

    A spatial method uses only the current minute of the neighbours, so one prediction over the
    window equals the scenario-by-scenario predictions; the result is cut into the scenarios of
    make_scenarios and has the same items as run_method. Temporal methods must use run_method.
    """
    whole = Scenario(
        str(event), target, pd.Timestamp(start), pd.Timestamp(end), tuple(neighbours)
    )
    inputs, truth = split(whole, data, history_min=0)
    pred = method.predict(inputs)
    x, y = components(inputs.current, whole.neighbours)
    ok = ~(np.isnan(x) | np.isnan(y))
    t = coords.loc[target]
    neigh = coords.loc[list(whole.neighbours)]
    d = np.asarray(geo.distance_km(t.lat, t.lon, neigh.lat, neigh.lon), float)
    nn = np.where(ok, d[None, :], np.inf).min(axis=1)
    nn[~ok.any(axis=1)] = np.nan
    n_inputs = ok.sum(axis=1)
    tv = truth.values[["X", "Y"]]
    runs = []
    for sc in make_scenarios(
        event,
        start,
        end,
        [target, *neighbours],
        targets=[target],
        length_min=length_min,
        kp3h=kp3h,
    ):
        rows = (tv.index >= sc.start) & (tv.index < sc.end)
        runs.append(
            {
                "scenario": sc,
                "pred": pred.loc[rows].copy(),
                "truth": tv.loc[rows].copy(),
                "n_inputs": n_inputs[rows],
                "nn_km": nn[rows],
            }
        )
    return runs


def common_masks(*run_lists) -> list[pd.Series]:
    """Per scenario, the minutes where every run list has a prediction and a truth (hard rule 4)."""
    masks = []
    for items in zip(*run_lists, strict=True):
        if len({(r["scenario"].target, r["scenario"].start) for r in items}) != 1:
            raise ValueError("run lists are not aligned scenario by scenario")
        frames = [f for r in items for f in (r["truth"], r["pred"])]
        masks.append(metrics.common_mask(*frames))
    return masks


def score_runs(runs, *, masks=None, kp_threshold: float = 5, **labels) -> pd.DataFrame:
    """One row per scenario; `labels` fill method, reference, station_set, causal, git, cfg."""
    rows = []
    for i, r in enumerate(runs):
        mask = None if masks is None else masks[i]
        m = metrics.score(r["pred"], r["truth"], mask)
        used = metrics.common_mask(r["truth"], r["pred"]).to_numpy(bool)
        if mask is not None:
            used = used & mask.reindex(r["truth"].index, fill_value=False).to_numpy(
                bool
            )
        sc = r["scenario"]
        rows.append(
            {
                "event": sc.event,
                "target": sc.target,
                "scenario_start": sc.start,
                **labels,
                "n_inputs": float(np.median(r["n_inputs"][used]))
                if used.any()
                else np.nan,
                "nn_distance_km": float(np.nanmedian(r["nn_km"][used]))
                if used.any()
                else np.nan,
                **m,
                "kp_group": kp_group(sc.kp, kp_threshold),
            }
        )
    df = pd.DataFrame(rows)
    first = [c for c in RESULT_COLUMNS if c in df.columns]
    return df[first + [c for c in df.columns if c not in first]]


def loo_select(
    make_method, params, stations, data, start, end, *, coords
) -> pd.DataFrame:
    """Pooled horizontal RMSE over [start, end) when each station of `stations` is hidden in turn
    and predicted from the others, for every parameter value. Used on the validation period, so
    the evaluation target never enters the selection (hard rule 2)."""
    scen = make_scenarios("selection", start, end, stations)
    rows = []
    for p in params:
        df = score_runs(run_method(make_method(p), scen, data, coords=coords))
        n = df["n_scored"].sum()
        se = np.nansum(df["n_scored"] * df["rmse_h"] ** 2)
        rows.append({"param": p, "rmse_h": float(np.sqrt(se / n)), "n_scored": int(n)})
    return pd.DataFrame(rows)
