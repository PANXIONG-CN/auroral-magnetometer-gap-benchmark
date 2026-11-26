"""Shared steps of E008-E011: references, station configurations, parameter selection without the
target, and method runs scored on common minutes.

The locked test (scripts/run_locked_test.py) calls the same functions as the development
experiments, so the methods it evaluates are the ones frozen in git.

Parameter selection never sees the target (hard rule 2): IDW power, SECS epsilon and the temporal
parameters by leave-one-out over the input stations on the validation period, ridge alpha by
leave-one-UT-day-out on the training period. Results are cached per input-station set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, evaluate, events, geo, mainfield, reference, secs, stats, storms
from .baselines import IDW, NearestNeighbour, RidgeMap
from .io import igrf, image, kp
from .scenarios import Scenario, make_scenarios, split
from .temporal import KalmanSECS, ResidualCorrection

DAY = pd.Timedelta(days=1)
MINUTE = pd.Timedelta(minutes=1)
NOON = pd.Timedelta(hours=12)
SPATIAL = ("idw", "nearest", "ridge", "secs")


@dataclass
class Setup:
    """Everything fixed before any event is read."""

    cfg: dict
    files: list[str]
    coords: pd.DataFrame
    kp3h: pd.DataFrame
    poles: pd.DataFrame
    height_km: float
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    d_train: pd.DataFrame
    d_val: pd.DataFrame  # validation period plus the history needed by temporal methods
    train_median: pd.Series
    train_mid: pd.Timestamp
    cache: dict = field(default_factory=dict)


def read_days(days, files) -> pd.DataFrame:
    return pd.concat([image.read_range(d, d + DAY, files) for d in days])


def mid_time(days) -> pd.Timestamp:
    return pd.DatetimeIndex(days).mean() + NOON


def make_setup(cfg: dict) -> Setup:
    """cfg keys: file_station_set, secs {height_km, grid_lat, grid_lon}, reference
    {training_quiet_days_before, training_lookback_days, min_coverage}, selection grids, temporal
    {history_min, ...}, sparsify {...}."""
    files = config.station_set(cfg["file_station_set"])
    coords = geo.station_coords(files)
    kp3h, _ = kp.read()
    sc = cfg["secs"]
    poles = secs.pole_grid(tuple(sc["grid_lat"]), tuple(sc["grid_lon"]))
    ev = config.load("events")
    tr, va = ev["training"], ev["validation"]
    scr = events.screening_cfg()
    rc = cfg["reference"]
    t_quiet = storms.quiet_days(
        kp3h,
        rc["training_quiet_days_before"],
        rc["training_lookback_days"],
        kp_max=scr["quiet_kp_max"],
    )
    t0, t1 = pd.Timestamp(tr["start"]), pd.Timestamp(tr["end"])
    v0, v1 = pd.Timestamp(va["start"]), pd.Timestamp(va["end"])
    train_raw = image.read_range(t0, t1, files)
    # training and validation keep E007's reference (D19): all minutes of the training quiet
    # days, no coverage threshold (KEV, KIL and OUJ fall below 90 % on all three days)
    ref_train = reference.quiet_constant(
        train_raw, t_quiet, min_coverage=rc.get("training_min_coverage", 0.0)
    )
    hist = cfg["temporal"]["history_min"] * MINUTE
    val_raw = image.read_range(v0 - hist, v1, files)
    return Setup(
        cfg=cfg,
        files=files,
        coords=coords,
        kp3h=kp3h,
        poles=poles,
        height_km=sc["height_km"],
        val_start=v0,
        val_end=v1,
        d_train=reference.disturbance(train_raw, ref_train),
        d_val=reference.disturbance(val_raw, ref_train),
        train_median=reference.median_reference(train_raw),
        train_mid=t0 + (t1 - t0) / 2,
    )


# --- events and references ---------------------------------------------------------------------


def event_data(s: Setup, p: dict) -> dict:
    """Quiet days (D14), raw quiet-day data, and raw event data from history_min before the
    window. Reads through image.read_range, so locked events need locked_access()."""
    scr = events.screening_cfg()
    qd = storms.quiet_days(
        s.kp3h, p["start"], scr["lookback_days"], kp_max=scr["quiet_kp_max"]
    )
    hist = s.cfg["temporal"]["history_min"] * MINUTE
    return {
        "event": p,
        "quiet_days": qd,
        "pre": read_days(qd, s.files) if qd else None,
        "raw": image.read_range(p["start"] - hist, p["end"], s.files),
    }


def core_change(model, coords, t0, t1) -> pd.Series:
    core = mainfield.core_xyz(model, coords, [t0, t1])
    return (core.iloc[1] - core.iloc[0]).rename_axis(["station", "component"])


def event_references(s: Setup, ed: dict, kinds) -> dict:
    """References for one event: quiet_constant, decomposed, training_median and
    training_median_core. The main field is the IGRF generation released before the event."""
    p, qd, pre, raw = ed["event"], ed["quiet_days"], ed["pre"], ed["raw"]
    cov = s.cfg["reference"]["min_coverage"]
    t_event = p["start"] + (p["end"] - p["start"]) / 2
    version = igrf.causal_version(p["start"])
    model = s.cache.setdefault(("igrf", version), igrf.load(version))
    out = {}
    for kind in kinds:
        if kind == "quiet_constant":
            out[kind] = reference.quiet_constant(pre, qd, min_coverage=cov)
        elif kind == "decomposed":
            out[kind] = reference.decomposed_reference(
                pre,
                qd,
                raw.index,
                core_change(model, s.coords, mid_time(qd), t_event),
                min_coverage=cov,
                smooth_min=s.cfg["reference"]["quiet_variation_smooth_min"],
            )
        elif kind == "training_median":
            out[kind] = s.train_median
        elif kind == "training_median_core":
            out[kind] = s.train_median + core_change(
                model, s.coords, s.train_mid, t_event
            ).reindex(s.train_median.index)
        else:
            raise ValueError(f"unknown reference {kind!r}")
    return out


# --- station configurations --------------------------------------------------------------------


def neighbours_by_distance(s: Setup, target: str) -> list[tuple[str, float]]:
    t = s.coords.loc[target]
    others = [x for x in s.files if x != target]
    d = geo.distance_km(
        t.lat, t.lon, s.coords.loc[others, "lat"], s.coords.loc[others, "lon"]
    )
    return sorted(zip(others, np.asarray(d, float)), key=lambda x: x[1])


def configurations(s: Setup, target: str) -> list[dict]:
    """Input-station sets for one target (E009). nearest_removed: the k nearest neighbours are
    missing together (k = 0 is the full network); random: n stations drawn at random, a fixed
    number of distinct draws per n, seeded by the config and the target."""
    sp = s.cfg["sparsify"]
    ranked = neighbours_by_distance(s, target)
    dist = dict(ranked)
    out = []

    def add(name, family, k, inputs):
        out.append(
            {
                "config": name,
                "family": family,
                "k": k,
                "inputs": tuple(sorted(inputs)),
                "n_config": len(inputs),
                "nominal_nn_km": round(min(dist[x] for x in inputs), 1),
            }
        )

    for k in range(sp["max_removed"] + 1):
        inputs = [x for x, _ in ranked[k:]]
        if len(inputs) >= sp["min_inputs"]:
            add(f"near{k}", "nearest_removed", k, inputs)
    rng = np.random.default_rng([sp["seed"], s.files.index(target)])
    pool = [x for x, _ in ranked]
    for n in sp["random_sizes"]:
        seen = set()
        for _ in range(100 * sp["random_draws"]):
            if len(seen) == sp["random_draws"]:
                break
            pick = tuple(
                sorted(str(x) for x in rng.choice(pool, size=n, replace=False))
            )
            if pick in seen:
                continue
            seen.add(pick)
            add(f"rand{n}_{len(seen)}", "random", len(pool) - n, pick)
    return out


# --- parameter selection without the target -----------------------------------------------------


def _loo_errors(models: dict, stations, data, start, end) -> pd.DataFrame:
    """Pooled horizontal RMSE when each station is predicted from the others over [start, end),
    one whole-window prediction per station; all models on the same minutes."""
    se = dict.fromkeys(models, 0.0)
    n = 0
    for h in stations:
        sc = Scenario("selection", h, start, end, tuple(x for x in stations if x != h))
        inputs, truth = split(sc, data, history_min=0)
        tv = truth.values[["X", "Y"]].to_numpy(float)
        preds = {
            k: m.predict(inputs)[["X", "Y"]].to_numpy(float) for k, m in models.items()
        }
        ok = ~np.isnan(tv).any(axis=1)
        for pr in preds.values():
            ok &= ~np.isnan(pr).any(axis=1)
        for k, pr in preds.items():
            se[k] += float(((pr[ok] - tv[ok]) ** 2).sum())
        n += int(ok.sum())
    return pd.DataFrame(
        [
            {"param": k, "rmse_h": np.sqrt(v / n) if n else np.nan, "n": n}
            for k, v in se.items()
        ]
    )


def secs_model(s: Setup, eps: float) -> secs.SECS:
    key = ("secs", eps)
    if key not in s.cache:
        s.cache[key] = secs.SECS(s.coords, s.poles, s.height_km, eps)
    return s.cache[key]


def select_spatial(s: Setup, inputs, fallback=None) -> dict:
    """IDW power and SECS epsilon by leave-one-out over `inputs` on the validation period. When
    fewer than selection.min_loo_minutes minutes can be scored (stations without validation data),
    the selection of `fallback` (the target's full network) is used instead."""
    key = ("spatial", frozenset(inputs))
    if key not in s.cache:
        g = s.cfg["selection"]
        st = sorted(inputs)
        idw = _loo_errors(
            {p: IDW(s.coords, power=p) for p in g["idw_power"]},
            st,
            s.d_val,
            s.val_start,
            s.val_end,
        )
        sec = _loo_errors(
            {e: secs_model(s, e) for e in g["secs_epsilon"]},
            st,
            s.d_val,
            s.val_start,
            s.val_end,
        )
        enough = min(idw["n"].min(), sec["n"].min()) >= g.get("min_loo_minutes", 1440)
        s.cache[key] = {
            "idw": float(idw.loc[idw["rmse_h"].idxmin(), "param"]) if enough else None,
            "secs": float(sec.loc[sec["rmse_h"].idxmin(), "param"]) if enough else None,
            "idw_table": idw,
            "secs_table": sec,
            "source": "inputs" if enough else None,
        }
    out = s.cache[key]
    if out["source"] is None:
        if fallback is None or frozenset(fallback) == frozenset(inputs):
            raise ValueError(f"cannot select parameters on {sorted(inputs)}")
        fb = select_spatial(s, fallback)
        out = {**fb, "source": "fallback"}
    return out


def select_ridge(s: Setup, target: str, inputs) -> float:
    key = ("ridge_alpha", target, tuple(sorted(inputs)))
    if key not in s.cache:
        t = RidgeMap.day_cv(
            s.d_train, target, tuple(sorted(inputs)), s.cfg["selection"]["ridge_alpha"]
        )
        s.cache[key] = float(t.loc[t["rmse_h"].idxmin(), "param"])
    return s.cache[key]


def ridge_model(s: Setup, target: str, inputs) -> RidgeMap:
    alpha = select_ridge(s, target, inputs)
    key = ("ridge", target, tuple(sorted(inputs)), alpha)
    if key not in s.cache:
        s.cache[key] = RidgeMap(alpha=alpha, dynamic=True).fit(
            s.d_train, target, tuple(sorted(inputs))
        )
    return s.cache[key]


def full_network(s: Setup, target: str) -> list[str]:
    return [x for x in s.files if x != target]


def spatial_models(s: Setup, target: str, inputs, methods=SPATIAL) -> dict:
    sel = select_spatial(s, inputs, fallback=full_network(s, target))
    make = {
        "idw": lambda: IDW(s.coords, power=sel["idw"]),
        "nearest": lambda: NearestNeighbour(s.coords),
        "ridge": lambda: ridge_model(s, target, inputs),
        "secs": lambda: secs_model(s, sel["secs"]),
    }
    return {m: make[m]() for m in methods}


def select_temporal(s: Setup, inputs, fallback=None) -> dict:
    """Bias-correction window and Kalman (a, lam) by leave-one-out over `inputs` on the
    validation period in 180-minute gaps with history (target never involved); `fallback` as in
    select_spatial."""
    key = ("temporal", frozenset(inputs))
    if key in s.cache:
        out = s.cache[key]
        if out["source"] is None:
            if fallback is None or frozenset(fallback) == frozenset(inputs):
                raise ValueError(
                    f"cannot select temporal parameters on {sorted(inputs)}"
                )
            return {**select_temporal(s, fallback), "source": "fallback"}
        return out
    tc = s.cfg["temporal"]
    eps = select_spatial(s, inputs, fallback=fallback)["secs"]
    base = secs_model(s, eps)
    windows = list(tc["bias_window_min"])
    grid = [(a, lam) for a in tc["kalman_a"] for lam in tc["kalman_lam"]]
    kf = KalmanSECS(
        s.coords,
        s.poles,
        s.height_km,
        a=[g[0] for g in grid],
        lam=[g[1] for g in grid],
    )
    keys = [("w", w) for w in windows] + [("k", g) for g in grid]
    se = dict.fromkeys(keys, 0.0)
    n = 0
    st = sorted(inputs)
    for h in st:
        neigh = [x for x in st if x != h]
        for sc in make_scenarios(
            "selection", s.val_start, s.val_end, [h, *neigh], targets=[h]
        ):
            inp, truth = split(sc, s.d_val, history_min=tc["history_min"])
            tv = truth.values[["X", "Y"]].to_numpy(float)
            if np.isnan(tv).any(axis=1).all():
                continue  # no truth for this station and gap (KIL, OUJ have no validation data)
            b = base.predict(inp)[["X", "Y"]].to_numpy(float)
            preds = {}
            for w in windows:
                corr = ResidualCorrection(base, window_min=w).correction(inp)
                preds[("w", w)] = b + corr
            kout = kf.run(inp)
            for i, g in enumerate(grid):
                preds[("k", g)] = kout[i]
            ok = ~np.isnan(tv).any(axis=1)
            for pr in preds.values():
                ok &= ~np.isnan(pr).any(axis=1)
            for k, pr in preds.items():
                se[k] += float(((pr[ok] - tv[ok]) ** 2).sum())
            n += int(ok.sum())
    table = pd.DataFrame(
        [
            {
                "kind": k[0],
                "param": str(k[1]),
                "rmse_h": np.sqrt(v / n) if n else np.nan,
                "n": n,
            }
            for k, v in se.items()
        ]
    )
    enough = n >= s.cfg["selection"].get("min_loo_minutes", 1440)
    best_w = min(windows, key=lambda w: se[("w", w)])
    best_k = min(grid, key=lambda g: se[("k", g)])
    s.cache[key] = {
        "window_min": best_w,
        "kalman": best_k,
        "secs": eps,
        "table": table,
        "source": "inputs" if enough else None,
    }
    return select_temporal(s, inputs, fallback)


def temporal_models(s: Setup, target: str, inputs) -> dict:
    sel = select_temporal(s, inputs, fallback=full_network(s, target))
    base = secs_model(s, sel["secs"])
    a, lam = sel["kalman"]
    return {
        "secs": base,
        "persistence": ResidualCorrection(
            base,
            window_min=s.cfg["temporal"]["persistence_window_min"],
            statistic="last",
        ),
        "bias_correction": ResidualCorrection(base, window_min=sel["window_min"]),
        "kalman": KalmanSECS(s.coords, s.poles, s.height_km, a=a, lam=lam),
    }


# --- runs and scoring ---------------------------------------------------------------------------


def runs_for(
    s: Setup, event: dict, data: pd.DataFrame, target: str, inputs, models: dict
) -> dict:
    """Run lists per model label over the event window. Spatial models predict the whole window
    at once; temporal ones (persistence, bias_correction, kalman) go scenario by scenario with
    history."""
    out = {}
    hist = s.cfg["temporal"]["history_min"]
    for label, m in models.items():
        name = getattr(m, "name", label)
        if name in ("persistence", "bias_correction", "kalman"):
            scen = make_scenarios(
                event["id"],
                event["start"],
                event["end"],
                [target, *inputs],
                targets=[target],
                kp3h=s.kp3h,
            )
            out[label] = evaluate.run_method(
                m, scen, data, coords=s.coords, history_min=hist
            )
        else:
            out[label] = evaluate.run_window(
                m,
                event["id"],
                event["start"],
                event["end"],
                target,
                inputs,
                data,
                coords=s.coords,
                kp3h=s.kp3h,
            )
    return out


def score_common(runs: dict, label_name: str = "method", **labels) -> pd.DataFrame:
    """Score every run list on the minutes common to all of them (hard rule 4)."""
    masks = evaluate.common_masks(*runs.values())
    frames = [
        evaluate.score_runs(r, masks=masks, **{label_name: k}, **labels)
        for k, r in runs.items()
    ]
    return pd.concat(frames, ignore_index=True)


METRICS = ["rmse_h", "rmse_x", "rmse_y", "peak_err", "peak_dt_abs_min", "dbdt_rmse"]


def summary_table(
    ev_table: pd.DataFrame, by: list[str], metrics=METRICS
) -> pd.DataFrame:
    """Mean over events and event-level bootstrap interval, per group in `by`."""
    rows = []
    for key, g in ev_table.groupby(by, sort=True):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row["events"] = int(g["event"].nunique())
        for k in metrics:
            b = stats.bootstrap_mean(g[k].to_numpy())
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = b["mean"], b["lo"], b["hi"]
        rows.append(row)
    return pd.DataFrame(rows)


def paired_table(
    ev_table: pd.DataFrame, by: list[str], key: str, pairs, metric: str = "rmse_h"
) -> pd.DataFrame:
    """Event-level a - b of `metric` within each group of `by`, bootstrapped over events."""
    rows = []
    for gkey, g in ev_table.groupby(by, sort=True):
        base = dict(zip(by, gkey if isinstance(gkey, tuple) else (gkey,)))
        for a, b in pairs:
            if a not in set(g[key]) or b not in set(g[key]):
                continue
            d = stats.paired_difference(g, metric, a, b, key=key)
            rows.append(
                {
                    **base,
                    "comparison": f"{a} - {b}",
                    "mean": d["mean"],
                    "lo": d["lo"],
                    "hi": d["hi"],
                    "events": d["n"],
                    "n_positive": int((d["per_event"] > 0).sum()),
                    "per_event": " ".join(
                        f"{k}:{v:.2f}" for k, v in d["per_event"].items()
                    ),
                }
            )
    return pd.DataFrame(rows)
