"""E011 evaluation (task T10): every claim test on a list of events with the frozen methods and
rules of experiments/E011_locked_test/config.yaml.

`collect` runs the methods and returns scenario rows; `analyse` turns them into the tables and
verdicts of docs/01_research_design.md. scripts/run_locked_test.py calls both on the locked test
events inside locked_access(); with --dev it calls them on the development events to check the
runner end to end without reading a locked event.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import config, evaluate, pipeline, reference, sparsity, stats
from .io import dst, igrf
from .temporal import KalmanSECS

TEMPORAL = ["persistence", "bias_correction", "kalman"]


def sparse_config(s, target, d_star, min_inputs):
    """The k nearest neighbours removed, smallest k with nearest-neighbour distance >= d_star."""
    ranked = pipeline.neighbours_by_distance(s, target)
    for k in range(len(ranked)):
        inputs = [x for x, _ in ranked[k:]]
        if len(inputs) < min_inputs:
            return None
        if ranked[k][1] >= d_star:
            return {
                "k": k,
                "inputs": tuple(sorted(inputs)),
                "nominal_nn_km": round(ranked[k][1], 1),
            }
    return None


def offset_errors(runs, masks, edges) -> pd.DataFrame:
    """Squared-error sums and minute counts per method and offset bin within the gap."""
    rows = []
    for label, rl in runs.items():
        for r, mask in zip(rl, masks, strict=True):
            ok = mask.to_numpy(bool)
            diff = r["pred"][["X", "Y"]].to_numpy() - r["truth"][["X", "Y"]].to_numpy()
            e2 = (diff**2).sum(axis=1)
            b = np.digitize(np.arange(len(e2)), edges[1:-1])
            for i in range(len(edges) - 1):
                use = ok & (b == i)
                rows.append(
                    {
                        "event": r["scenario"].event,
                        "method": label,
                        "offset_lo": edges[i],
                        "se": float(e2[use].sum()),
                        "n": int(use.sum()),
                    }
                )
    return pd.DataFrame(rows)


def bias_table(s, refs: dict, p: dict, quiet_days) -> pd.DataFrame:
    """Per station: quiet constant minus training median minus the IGRF core change between the
    training period and the quiet days, i.e. the change of station bias (X, Y)."""
    model = s.cache[("igrf", igrf.causal_version(p["start"]))]
    qc = refs["quiet_constant"]
    core = pipeline.core_change(
        model, s.coords, s.train_mid, pipeline.mid_time(quiet_days)
    )
    d = (qc - s.train_median - core.reindex(qc.index)).unstack("component")[["X", "Y"]]
    return d.rename(columns={"X": "bias_change_X", "Y": "bias_change_Y"}).assign(
        event=p["id"]
    )


def collect(cfg: dict, periods, *, git: str, cfg_hash: str, log=print) -> dict:
    s = pipeline.make_setup(cfg)
    targets = config.station_set(cfg["targets"])
    d_star = cfg["d_star_km"]
    kinds = list(dict.fromkeys(["quiet_constant", *cfg["reference"]["c1_kinds"]]))
    labels = {"causal": True, "git": git, "cfg": cfg_hash}
    confs = {t: pipeline.configurations(s, t) for t in targets}
    sparse = {
        t: sparse_config(s, t, d_star, cfg["sparsify"]["min_inputs"]) for t in targets
    }
    wf_cfg = cfg["waterfall"]
    f4 = cfg["f4"]
    out = {
        k: []
        for k in (
            "c1",
            "c1_clean",
            "c2",
            "c3",
            "offsets",
            "waterfall",
            "bias",
            "f4",
            "configs",
        )
    }
    for t in targets:
        for c in confs[t]:
            out["configs"].append(
                {
                    "target": t,
                    **{k: v for k, v in c.items() if k != "inputs"},
                    "inputs": " ".join(c["inputs"]),
                }
            )
    for p in periods:
        ed = pipeline.event_data(s, p)
        refs = pipeline.event_references(s, ed, kinds)
        dist = {k: reference.disturbance(ed["raw"], r) for k, r in refs.items()}
        bias = bias_table(s, refs, p, ed["quiet_days"])
        flagged = flag_stations(bias, cfg["c1"]["bias_threshold_nT"])
        bias["flagged"] = bias.index.isin(flagged)
        out["bias"].append(bias)
        for t in targets:
            full = tuple(sorted(pipeline.full_network(s, t)))
            sm = pipeline.spatial_models(s, t, full, cfg["methods"])
            for m in cfg["c1"]["methods"]:
                runs = {
                    k: pipeline.runs_for(s, p, dist[k], t, full, {m: sm[m]})[m]
                    for k in cfg["reference"]["c1_kinds"]
                }
                out["c1"].append(
                    pipeline.score_common(
                        runs,
                        label_name="reference",
                        method=m,
                        station_set="loo11",
                        **labels,
                    )
                )
            clean = tuple(x for x in full if x not in flagged)
            if (
                flagged
                and t not in flagged
                and len(clean) >= cfg["sparsify"]["min_inputs"]
            ):
                smc = pipeline.spatial_models(s, t, clean, cfg["c1"]["methods"])
                for m in cfg["c1"]["methods"]:
                    runs = {
                        k: pipeline.runs_for(s, p, dist[k], t, clean, {m: smc[m]})[m]
                        for k in ("training_median", "decomposed")
                    }
                    out["c1_clean"].append(
                        pipeline.score_common(
                            runs,
                            label_name="reference",
                            method=m,
                            station_set="clean",
                            **labels,
                        )
                    )
            for c in confs[t]:
                models = pipeline.spatial_models(s, t, c["inputs"], cfg["methods"])
                runs = pipeline.runs_for(
                    s, p, dist["quiet_constant"], t, c["inputs"], models
                )
                out["c2"].append(
                    pipeline.score_common(
                        runs,
                        reference="quiet_constant",
                        station_set=c["config"],
                        **labels,
                    ).assign(
                        config=c["config"],
                        family=c["family"],
                        k=c["k"],
                        n_config=c["n_config"],
                        nominal_nn_km=c["nominal_nn_km"],
                        inside_hull=sparsity.inside_hull(s.coords, t, c["inputs"]),
                    )
                )
            nn0 = pipeline.neighbours_by_distance(s, t)[0][1]
            dense = {"inputs": full, "k": 0} if nn0 < d_star else None
            for cls, conf in [("dense", dense), ("sparse", sparse[t])]:
                if conf is None:
                    continue
                inputs = conf["inputs"]
                tm = pipeline.temporal_models(s, t, inputs)
                tm["kalman_static"] = KalmanSECS(
                    s.coords, s.poles, s.height_km, a=0.0, lam=tm["kalman"].lam[0]
                )
                runs = pipeline.runs_for(s, p, dist["quiet_constant"], t, inputs, tm)
                masks = evaluate.common_masks(*runs.values())
                out["c3"].append(
                    pipeline.score_common(
                        runs, reference="quiet_constant", station_set=cls, **labels
                    ).assign(k=conf["k"])
                )
                out["offsets"].append(
                    offset_errors(
                        runs, masks, cfg["temporal"]["offset_bins_min"]
                    ).assign(target=t, station_set=cls)
                )
                sm_c = pipeline.spatial_models(s, t, inputs, ["idw", "secs", "ridge"])
                step_runs = {}
                for st in wf_cfg["steps"]:
                    model = (
                        tm[wf_cfg["temporal_method"]]
                        if st["method"] == "temporal"
                        else sm_c[st["method"]]
                    )
                    step_runs[st["label"]] = pipeline.runs_for(
                        s, p, dist[st["reference"]], t, inputs, {"m": model}
                    )["m"]
                out["waterfall"].append(
                    pipeline.score_common(
                        step_runs, label_name="step", station_set=cls, **labels
                    ).assign(k=conf["k"])
                )
                if (
                    cls == "dense"
                    and t == f4["target"]
                    and p["id"] in (f4["event"], f4.get("dev_event"))
                ):
                    series = {}
                    for label, rl in {
                        **{
                            m: pipeline.runs_for(
                                s, p, dist["quiet_constant"], t, full, {m: sm[m]}
                            )[m]
                            for m in ("secs", "idw", "ridge")
                        },
                        wf_cfg["temporal_method"]: runs[wf_cfg["temporal_method"]],
                    }.items():
                        series[(label, "X")] = pd.concat([r["pred"]["X"] for r in rl])
                        series[(label, "Y")] = pd.concat([r["pred"]["Y"] for r in rl])
                    series[("observed", "X")] = pd.concat(
                        [r["truth"]["X"] for r in runs["secs"]]
                    )
                    series[("observed", "Y")] = pd.concat(
                        [r["truth"]["Y"] for r in runs["secs"]]
                    )
                    d, _ = dst.read(p["start"], p["end"])
                    t_min = d.idxmin()
                    h = pd.Timedelta(hours=f4["half_window_h"])
                    f = pd.DataFrame(series)
                    f = f.loc[(f.index >= t_min - h) & (f.index < t_min + h)]
                    f.columns = [f"{a}_{c}" for a, c in f.columns]
                    f = f.rename_axis("time").reset_index()
                    out["f4"].append(
                        f.assign(event=p["id"], target=t, dst_min_time=t_min)
                    )
        log(f"{p['id']} done")
    res = {}
    for k, v in out.items():
        if k == "configs":
            res[k] = pd.DataFrame(v)
        elif v:
            res[k] = pd.concat(v, ignore_index=k != "bias")
    return res


def verdict_c2(tests: pd.DataFrame) -> dict:
    """Per comparison 'other - secs' (docs/01_research_design.md, C2):
    supported: slope and sparse-end difference both above 0;
    grows_sparse_end_not_significant: slope above 0, sparse-end interval includes 0;
    gain_shrinks_with_sparsity: slope below 0;
    gain_independent_of_density: slope interval includes 0."""
    v = {}
    for _, r in tests.iterrows():
        if r["slope_lo"] > 0 and r["sparse_end_lo"] > 0:
            v[r["comparison"]] = "supported"
        elif r["slope_lo"] > 0:
            v[r["comparison"]] = "grows_sparse_end_not_significant"
        elif r["slope_hi"] < 0:
            v[r["comparison"]] = "gain_shrinks_with_sparsity"
        else:
            v[r["comparison"]] = "gain_independent_of_density"
    return v


def verdict_c3(gains: pd.DataFrame) -> dict:
    """Per temporal method, from the gains 'secs - method' over all minutes (C3):
    supported: dense interval includes 0, sparse interval above 0;
    gain_in_both / gain_only_dense: intervals above 0 where named;
    worse_in_both: both intervals below 0; no_gain_in_either: otherwise."""
    v = {}
    g = gains[gains["kp_group"] == "all"]
    for m in TEMPORAL:
        rows = {
            cls: g[(g["comparison"] == f"secs - {m}") & (g["station_set"] == cls)]
            for cls in ("dense", "sparse")
        }
        if any(r.empty for r in rows.values()):
            continue
        d, sp = rows["dense"].iloc[0], rows["sparse"].iloc[0]
        if d["lo"] <= 0 <= d["hi"] and sp["lo"] > 0:
            v[m] = "supported"
        elif d["lo"] > 0 and sp["lo"] > 0:
            v[m] = "gain_in_both"
        elif d["lo"] > 0:
            v[m] = "gain_only_dense"
        elif d["hi"] < 0 and sp["hi"] < 0:
            v[m] = "worse_in_both"
        else:
            v[m] = "no_gain_in_either"
    return v


def flag_stations(bias: pd.DataFrame, threshold: float) -> list[str]:
    """Stations whose bias change, relative to the network median of the event, reaches the
    threshold in X or Y (same threshold as the step rule of D17). Adds the relative columns."""
    xy = bias[["bias_change_X", "bias_change_Y"]]
    rel = xy - xy.median()
    bias["relative_X"] = rel["bias_change_X"]
    bias["relative_Y"] = rel["bias_change_Y"]
    return sorted(rel.index[(rel.abs() >= threshold).any(axis=1)])


def verdict_c1(
    all_paired: pd.Series,
    clean_paired: pd.Series,
    *,
    any_flagged: bool,
    min_share: float = 0.5,
) -> dict:
    """C1 for one method (docs/01_research_design.md). all_paired: event-level
    'training_median - decomposed' on the full network and every target (mean, lo, n_positive,
    events); clean_paired: the same with the flagged stations removed as inputs and targets.

    supported: the decrease holds in every event, its interval excludes 0, and it does not hinge
    on the flagged stations; few_stations_driven: flagged stations exist and without them less
    than `min_share` of the decrease remains or its interval includes 0; not_all_events: the
    decrease fails in some event or its interval includes 0."""
    test = bool(
        all_paired["n_positive"] == all_paired["events"] and all_paired["lo"] > 0
    )
    share = (
        float(clean_paired["mean"] / all_paired["mean"])
        if all_paired["mean"]
        else float("nan")
    )
    few = bool(any_flagged and (share < min_share or clean_paired["lo"] <= 0))
    if few:
        v = "few_stations_driven"
    elif test:
        v = "supported"
    else:
        v = "not_all_events"
    return {"verdict": v, "all_events_decrease": test, "remaining_share": share}


def analyse(res: dict, cfg: dict) -> dict:
    """Tables and verdicts for C1-C3 and F2 from the scenario rows of `collect`."""
    out = {}
    # --- C1 ---------------------------------------------------------------------------------
    c1 = res["c1"]
    ev = stats.event_summary(c1, by=["event", "method", "reference"])
    out["c1_events"] = ev
    pairs = [
        ("training_median", "decomposed"),
        ("training_median", "training_median_core"),
        ("training_median_core", "quiet_constant"),
        ("quiet_constant", "decomposed"),
    ]
    out["c1_paired"] = pipeline.paired_table(ev, ["method"], "reference", pairs)
    bias = res["bias"]
    flagged = {e: set(g.index[g["flagged"]]) for e, g in bias.groupby("event")}
    tm_dec = c1[c1["reference"].isin(["training_median", "decomposed"])]
    clean_parts = []
    for e, g in tm_dec.groupby("event"):
        if flagged.get(e):
            clean_parts.append(res["c1_clean"][res["c1_clean"]["event"] == e])
        else:
            clean_parts.append(g)
    clean = pd.concat(clean_parts, ignore_index=True)
    ev_clean = stats.event_summary(clean, by=["event", "method", "reference"])
    out["c1_clean_events"] = ev_clean
    pc = pipeline.paired_table(
        ev_clean, ["method"], "reference", [("training_median", "decomposed")]
    )
    out["c1_clean_paired"] = pc
    out["c1_flagged"] = pd.DataFrame(
        [
            {"event": e, "flagged": " ".join(sorted(v))}
            for e, v in sorted(flagged.items())
        ]
    )
    verdict = {}
    pa = out["c1_paired"]
    for m in cfg["c1"]["methods"]:
        a_ = pa[
            (pa["method"] == m) & (pa["comparison"] == "training_median - decomposed")
        ].iloc[0]
        c_ = pc[pc["method"] == m].iloc[0]
        verdict[m] = verdict_c1(
            a_,
            c_,
            any_flagged=any(flagged.values()),
            min_share=cfg["c1"]["min_remaining_share"],
        )
    out["c1_verdict"] = verdict
    out["bias"] = bias
    # --- C2 ---------------------------------------------------------------------------------
    edges = cfg["bins"]["nn_km"]
    rows = sparsity.add_bins(res["c2"], edges)
    centres = sparsity.bin_centres(rows)
    ev_nn = stats.event_summary(rows, by=["event", "nn_bin", "method"])
    out["c2_curves"] = pipeline.summary_table(ev_nn, ["nn_bin", "method"]).assign(
        centre_km=lambda t: centres.reindex(t["nn_bin"]).to_numpy()
    )
    out["c2_paired"] = pipeline.paired_table(
        ev_nn,
        ["nn_bin"],
        "method",
        [("idw", "secs"), ("nearest", "secs"), ("ridge", "secs")],
    ).assign(centre_km=lambda t: centres.reindex(t["nn_bin"]).to_numpy())
    ev_n = stats.event_summary(rows, by=["event", "n_inputs_int", "method"])
    out["c2_curves_n"] = pipeline.summary_table(ev_n, ["n_inputs_int", "method"])
    ev_h = stats.event_summary(rows, by=["event", "inside_hull", "method"])
    out["c2_hull"] = pipeline.paired_table(
        ev_h, ["inside_hull"], "method", [("idw", "secs"), ("ridge", "secs")]
    )
    far = rows[rows["nn_distance_km"] >= cfg["d_star_km"]]
    tests = sparsity.c2_tests(ev_nn, centres, far, ["idw", "nearest", "ridge"])
    out["c2_tests"] = tests
    excl = cfg.get("gate_a_refit_exclude", [])
    rel = sparsity.relative_error(rows, exclude=excl)
    out["c2_relative"] = rel
    kept = rows[~rows["target"].isin(excl) & (rows["config"] != "near0")]
    out["c2_gate_a_refit"] = sparsity.gate_a(rel, sparsity.bin_centres(kept), edges)
    out["c2_verdict"] = verdict_c2(tests)
    out["c2_mean_gain"] = pipeline.paired_table(
        stats.event_summary(rows, by=["event", "method"]).assign(all="all"),
        ["all"],
        "method",
        [("idw", "secs"), ("ridge", "secs")],
    )
    # --- C3 ---------------------------------------------------------------------------------
    c3 = res["c3"]
    ev3 = []
    for group, g in [("all", c3), ("Kp>=5", c3[c3["kp_group"] == "Kp>=5"])]:
        ev3.append(
            stats.event_summary(g, by=["event", "station_set", "method"]).assign(
                kp_group=group
            )
        )
    ev3 = pd.concat(ev3, ignore_index=True)
    out["c3_events"] = ev3
    out["c3_summary"] = pipeline.summary_table(
        ev3, ["kp_group", "station_set", "method"]
    )
    gains = pipeline.paired_table(
        ev3,
        ["kp_group", "station_set"],
        "method",
        [("secs", m) for m in TEMPORAL]
        + [("secs", "kalman_static"), ("kalman_static", "kalman")],
    )
    out["c3_gains"] = gains
    out["c3_verdict"] = verdict_c3(gains)
    o = (
        res["offsets"]
        .groupby(["event", "station_set", "method", "offset_lo"])[["se", "n"]]
        .sum()
        .reset_index()
    )
    o["rmse_h"] = np.sqrt(o["se"] / o["n"])
    og = []
    for (cls, lo), gg in o.groupby(["station_set", "offset_lo"]):
        w = gg.pivot(index="event", columns="method", values="rmse_h")
        for m in TEMPORAL:
            b = stats.bootstrap_mean((w["secs"] - w[m]).to_numpy())
            og.append(
                {
                    "station_set": cls,
                    "offset_lo": lo,
                    "method": m,
                    "gain": b["mean"],
                    "lo": b["lo"],
                    "hi": b["hi"],
                    "secs_rmse": float(w["secs"].mean()),
                }
            )
    out["c3_offsets"] = pd.DataFrame(og)
    # --- F2 waterfall -----------------------------------------------------------------------
    wf = res["waterfall"]
    evw = stats.event_summary(wf, by=["event", "station_set", "step"])
    out["waterfall_events"] = evw
    out["waterfall_summary"] = pipeline.summary_table(evw, ["station_set", "step"])
    steps = [s["label"] for s in cfg["waterfall"]["steps"]]
    out["waterfall_steps"] = pipeline.paired_table(
        evw, ["station_set"], "step", list(itertools.pairwise(steps))
    )
    out["f4"] = res.get("f4")
    return out
