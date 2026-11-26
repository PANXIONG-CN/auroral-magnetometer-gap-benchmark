"""Paper figures F1-F5 (docs/01_research_design.md, "论文关键图").

    uv run scripts/make_figures.py f1            # station network and event timeline
    uv run scripts/make_figures.py f3 f5 ...     # from experiment results (paths in FIGURES)

F1 needs only the configuration. The others read result tables; each figure names its source in
the caption file written next to it (figures/F?_caption.txt).
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

from p1 import config, events, geo, paths

OUT = paths.ROOT / "figures"
PAPER = False
METHOD_LABELS = {
    "secs": "SECS",
    "ridge": "Ridge",
    "idw": "IDW",
    "nearest": "Nearest",
    "kalman": "Kalman SECS",
    "observed": "Observed",
}
COMP_LABELS = {
    "idw - secs": "IDW \u2212 SECS",
    "ridge - secs": "Ridge \u2212 SECS",
}
plt.rcParams.update(
    {"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8}
)


def f1() -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    files = config.station_set("leave_one_out_12")
    six = set(config.station_set("original_6"))
    c = geo.station_coords(files)
    proj = ccrs.LambertConformal(
        central_longitude=22, central_latitude=67, standard_parallels=(64, 70)
    )
    fig = plt.figure(figsize=(7.2, 8.0))
    ax = fig.add_axes([0.08, 0.35, 0.86, 0.6], projection=proj)
    ax.set_extent([12, 32, 63.5, 71.2], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="0.93")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, linestyle=":")
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="0.6", x_inline=False, y_inline=False
    )
    gl.top_labels = gl.right_labels = False
    others = c.drop(index="SOD")
    pts = np.column_stack([others.lon, others.lat])
    hull = ConvexHull(pts)
    ring = np.vstack([pts[hull.vertices], pts[hull.vertices[:1]]])
    ax.fill(
        ring[:, 0],
        ring[:, 1],
        transform=ccrs.Geodetic(),
        color="tab:blue",
        alpha=0.12,
        lw=0,
    )
    ax.plot(ring[:, 0], ring[:, 1], transform=ccrs.Geodetic(), color="tab:blue", lw=0.8)
    for st, r in c.iterrows():
        kw = {"marker": "^", "ms": 8, "mec": "k", "mew": 0.6}
        color = "tab:red" if st == "SOD" else ("tab:orange" if st in six else "white")
        ax.plot(r.lon, r.lat, color=color, transform=ccrs.PlateCarree(), **kw)
        ax.text(
            r.lon + 0.25, r.lat + 0.08, st, transform=ccrs.PlateCarree(), fontsize=8
        )
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="^",
                ls="",
                color="tab:red",
                mec="k",
                label="SOD (first target)",
            ),
            Line2D(
                [],
                [],
                marker="^",
                ls="",
                color="tab:orange",
                mec="k",
                label="original 6 inputs",
            ),
            Line2D(
                [],
                [],
                marker="^",
                ls="",
                color="white",
                mec="k",
                label="added in the 11-station set",
            ),
            Line2D(
                [], [], color="tab:blue", label="convex hull of the 11 inputs for SOD"
            ),
        ],
        loc="lower left",
    )
    ax.set_title("(a) IMAGE stations; each of the 12 is hidden in turn")

    ax2 = fig.add_axes([0.2, 0.06, 0.76, 0.2])
    ev = config.load("events")
    groups = [
        (
            "training + validation",
            [
                (
                    pd.Timestamp(ev["training"]["start"]),
                    pd.Timestamp(ev["validation"]["end"]),
                )
            ],
            "tab:green",
        ),
        (
            "development",
            [(p["start"], p["end"]) for p in events.fixed_periods("development")],
            "tab:grey",
        ),
        (
            "locked test",
            [(pd.Timestamp(e["start"]), pd.Timestamp(e["end"])) for e in ev["test"]],
            "tab:red",
        ),
    ]
    for i, (name, spans, color) in enumerate(groups):
        for a, b in spans:
            ax2.plot([a, b], [i, i], lw=8, color=color, solid_capstyle="butt")
            ax2.plot([a], [i], marker="|", ms=14, color=color)
    previous = None
    for e in sorted(ev["test_candidates"], key=lambda e: e["start"]):
        t = pd.Timestamp(e["start"])
        close = previous is not None and (t - previous).days < 160
        ax2.annotate(
            f"{e['dst_min']}",
            (t, 2),
            xytext=(0, 19 if close else 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="tab:red",
        )
        previous = t
    ax2.set_yticks(range(len(groups)), [g[0] for g in groups])
    ax2.set_ylim(-0.7, 3.3)
    ax2.set_xlim(pd.Timestamp("2014-10-01"), pd.Timestamp("2025-03-01"))
    ax2.set_title(
        "(b) periods (5-day storm windows); red numbers: Dst minimum of each locked storm (nT)"
    )
    ax2.grid(axis="x", lw=0.3)
    fig.savefig(OUT / "F1_network_design.png", dpi=300)
    fig.savefig(OUT / "F1_network_design.pdf")
    if not PAPER:
        (OUT / "F1_caption.txt").write_text(
            "Figure 1. Station network and evaluation design. (a) The 12 IMAGE magnetometer stations. "
            "Each station is hidden in turn and reconstructed from the other 11; the shaded polygon is "
            "the convex hull of the 11 inputs used for SOD. (b) Training (2017-08-01 to 07), validation "
            "(2017-08-08 to 10), the three development storms and the nine locked test storms, each a "
            "5-day window starting on the UT day of the Dst main-phase onset.\n",
            encoding="utf-8",
        )


E011 = paths.ROOT / "experiments" / "E011_locked_test"
COLORS = {
    "secs": "tab:blue",
    "ridge": "tab:green",
    "idw": "tab:orange",
    "nearest": "tab:grey",
    "persistence": "tab:purple",
    "bias_correction": "tab:brown",
    "kalman": "tab:red",
    "kalman_static": "tab:pink",
    "observed": "k",
}
SOURCE = {"dir": E011 / "results", "tag": "locked"}


def table(name: str) -> pd.DataFrame:
    """Latest E011 table `name` from the chosen source (locked results or the dev check)."""
    files = sorted(SOURCE["dir"].glob(f"{name}_{SOURCE['tag']}_*.csv"))
    if not files:
        raise FileNotFoundError(f"no {name}_{SOURCE['tag']}_*.csv in {SOURCE['dir']}")
    return pd.read_csv(files[-1], dtype={"event": str})


def note() -> str:
    return (
        "" if SOURCE["tag"] == "locked" else " [development check, not the locked test]"
    )


def ci(ax, x, g, col="rmse_h", **kw):
    ax.errorbar(
        x,
        g[col],
        yerr=[g[col] - g[f"{col}_lo"], g[f"{col}_hi"] - g[col]],
        capsize=3,
        **kw,
    )


def f2() -> None:
    summ = table("waterfall_summary")
    steps = table("waterfall_steps")
    order = list(dict.fromkeys(summ["step"]))

    cfg = __import__("yaml").safe_load(
        (E011 / "config.yaml").read_text(encoding="utf-8")
    )
    order = [s["label"] for s in cfg["waterfall"]["steps"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, cls in zip(axes, ["dense", "sparse"], strict=True):
        g = summ[summ["station_set"] == cls].set_index("step").reindex(order)
        x = np.arange(len(order))
        ax.bar(x, g["rmse_h"], color=["0.6", "0.75", "tab:blue", "tab:red"], width=0.6)
        ax.errorbar(
            x,
            g["rmse_h"],
            yerr=[g["rmse_h"] - g["rmse_h_lo"], g["rmse_h_hi"] - g["rmse_h"]],
            fmt="none",
            ecolor="k",
            capsize=3,
            lw=0.8,
        )
        st = steps[steps["station_set"] == cls].set_index("comparison")
        for i in range(1, len(order)):
            key = f"{order[i - 1]} - {order[i]}"
            if key in st.index:
                d = st.loc[key]
                ax.annotate(
                    f"{-d['mean']:+.1f}\n[{-d['hi']:+.1f}, {-d['lo']:+.1f}]",
                    (i, g["rmse_h_hi"].iloc[i]),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )
        ax.set_xticks(x, [o.replace(", ", ",\n") for o in order], fontsize=7)
        ax.set_title(f"({'ab'[cls == 'sparse']}) {cls} configuration")
    axes[0].set_ylabel("horizontal RMSE (nT, log scale)")
    axes[0].set_yscale("log")
    axes[0].set_ylim(10, 2.0 * float(summ["rmse_h_hi"].max()))
    if not PAPER:
        fig.suptitle("Error budget, all 12 stations hidden in turn" + note(), fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "F2_error_budget.png", dpi=300)
    fig.savefig(OUT / "F2_error_budget.pdf")


def f3() -> None:
    cur = table("c2_curves")
    par = table("c2_paired")
    rel = table("c2_relative")
    curn = table("c2_curves_n")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2 if PAPER else 6.4))
    ax = axes[0, 0]
    for m in ["secs", "ridge", "idw", "nearest"]:
        g = cur[cur["method"] == m].sort_values("centre_km")
        lab = METHOD_LABELS[m] if PAPER else m
        ci(ax, g["centre_km"], g, marker="o", ms=3, label=lab, color=COLORS[m])
    ax.set_xlabel("distance to nearest valid neighbour (km)")
    ax.set_ylabel("horizontal RMSE (nT)")
    ax.legend()
    ax.set_title("(a) error against nearest-neighbour distance")
    ax = axes[0, 1]
    for comp in ["idw - secs", "ridge - secs"]:
        g = par[par["comparison"] == comp].sort_values("centre_km")
        lab = COMP_LABELS[comp] if PAPER else comp
        ax.errorbar(
            g["centre_km"],
            g["mean"],
            yerr=[g["mean"] - g["lo"], g["hi"] - g["mean"]],
            marker="o",
            ms=3,
            capsize=3,
            label=lab,
            color=COLORS[comp.split(" - ")[0]],
        )
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xlabel("distance to nearest valid neighbour (km)")
    ax.set_ylabel("RMSE difference (nT), > 0: SECS better")
    ax.legend()
    ax.set_title("(b) other method minus SECS")
    ax = axes[1, 0]
    w = rel.pivot(index="nn_bin", columns="event", values="relative")
    cen = cur.drop_duplicates("nn_bin").set_index("nn_bin")["centre_km"]
    w = w.loc[[b for b in cen.sort_values().index if b in w.index]]
    m = w.mean(axis=1)
    ax.plot(cen.reindex(w.index), m, "o-", color=COLORS["secs"])
    ax.set_xlabel("distance to nearest valid neighbour (km)")
    ax.set_ylabel("SECS RMSE / full-network RMSE")
    ax.set_title("(c) SECS error relative to the full network")
    ax = axes[1, 1]
    for mth in ["secs", "ridge", "idw", "nearest"]:
        g = curn[(curn["method"] == mth) & (curn["n_inputs_int"] >= 3)].sort_values(
            "n_inputs_int"
        )
        lab = METHOD_LABELS[mth] if PAPER else mth
        ci(ax, g["n_inputs_int"], g, marker="o", ms=3, label=lab, color=COLORS[mth])
    ax.set_xlabel("number of valid input stations")
    ax.set_ylabel("horizontal RMSE (nT)")
    ax.set_title("(d) error against number of inputs")
    if not PAPER:
        fig.suptitle("Network sparsity; bars: 95% event bootstrap" + note(), fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "F3_sparsity.png", dpi=300)
    fig.savefig(OUT / "F3_sparsity.pdf")


def f4() -> None:
    f = table("f4")
    f.index = pd.to_datetime(f["time"])
    labels = [c[:-2] for c in f.columns if c.endswith("_X")]
    first = ["observed", "idw", "ridge", "secs"]
    order = first + [x for x in labels if x not in first]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.8 if PAPER else 7.0), sharex=True)
    for lab in [x for x in order if x in labels]:
        disp = METHOD_LABELS.get(lab, lab) if PAPER else lab
        kw = {
            "color": COLORS.get(lab, "k"),
            "lw": 1.2 if lab == "observed" else 0.8,
            "label": disp,
        }
        axes[0].plot(f.index, f[f"{lab}_X"], **kw)
        axes[1].plot(f.index, f[f"{lab}_Y"], **kw)
        axes[2].plot(
            f.index, np.hypot(f[f"{lab}_X"].diff(), f[f"{lab}_Y"].diff()), **kw
        )
    axes[0].set_ylabel("ΔX (nT)")
    axes[1].set_ylabel("ΔY (nT)")
    axes[2].set_ylabel("|dB_H/dt| (nT/min)")
    axes[0].legend(ncol=3, fontsize=7)
    if PAPER:
        t0 = f.index.min().floor("3h")
        boundaries = pd.date_range(t0, f.index.max(), freq="3h")
        for b in boundaries:
            if f.index.min() < b < f.index.max():
                for ax in axes:
                    ax.axvline(b, color="0.55", lw=0.5, ls=":")
        axes[0].set_title("(a) northward disturbance ΔX", fontsize=8.5)
        axes[1].set_title("(b) eastward disturbance ΔY", fontsize=8.5)
        axes[2].set_title("(c) horizontal rate |dB_H/dt|", fontsize=8.5)
        axes[2].set_xlabel("UTC (2024-05-10 to 2024-05-11)")
    else:
        title = f"{f['target'].iloc[0]} hidden, storm {f['event'].iloc[0]}, 12 h around the Dst minimum"
        axes[0].set_title(title + note(), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "F4_storm_example.png", dpi=300)
    fig.savefig(OUT / "F4_storm_example.pdf")


def f5() -> None:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib import patheffects

    from p1 import stats

    rows = table("c2")
    rows = rows[rows["config"] == "near0"]
    ev = stats.event_summary(rows, by=["event", "target", "method"])
    mean = ev.groupby(["target", "method"])["rmse_h"].mean().unstack()
    files = config.station_set("leave_one_out_12")
    c = geo.station_coords(files)
    proj = ccrs.LambertConformal(
        central_longitude=22, central_latitude=67, standard_parallels=(64, 70)
    )
    halo = [patheffects.withStroke(linewidth=2.5, foreground="white")]
    fig = plt.figure(figsize=(7.2, 3.4 if PAPER else 3.6))
    panels = [
        (
            "secs",
            "SECS RMSE when hidden (nT)",
            "viridis",
            {"vmin": 10, "vmax": 45},
            "max",
        ),
        ("diff", "IDW − SECS (nT); > 0: SECS better", "RdBu_r", None, "neither"),
    ]
    h_frac = 0.88 if PAPER else 0.84
    for i, (col, lab, cmap, lim, extend) in enumerate(panels):
        ax = fig.add_axes([0.01 + 0.5 * i, 0.04, 0.4, h_frac], projection=proj)
        ax.set_extent([13.5, 31, 63.9, 70.5], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="0.93")
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.4)
        v = mean["secs"] if col == "secs" else mean["idw"] - mean["secs"]
        if lim is None:
            m = float(np.nanmax(np.abs(v)))
            lim = {"vmin": -m, "vmax": m}
        sc = ax.scatter(
            c.lon,
            c.lat,
            c=v.reindex(c.index),
            s=70,
            edgecolor="k",
            lw=0.6,
            transform=ccrs.PlateCarree(),
            zorder=3,
            cmap=cmap,
            **lim,
        )
        for st, r in c.iterrows():
            left = st == "AND"  # TRO sits right of AND
            ax.text(
                r.lon + (-0.45 if left else 0.45),
                r.lat + 0.12,
                f"{st} {v.get(st, np.nan):.0f}",
                fontsize=6.5,
                ha="right" if left else "left",
                transform=ccrs.PlateCarree(),
                zorder=4,
                path_effects=halo,
            )
        fig.colorbar(sc, ax=ax, shrink=0.85, label=lab, extend=extend)
        ax.set_title(f"({'ab'[i]}) " + lab.split(" (")[0], fontsize=8)
    if not PAPER:
        fig.suptitle(
            "Each station hidden in turn, full network, mean over the nine locked storms"
            + note(),
            fontsize=8.5,
        )
    save_kw = {"bbox_inches": "tight", "pad_inches": 0.04} if PAPER else {}
    fig.savefig(OUT / "F5_station_map.png", dpi=300, **save_kw)
    fig.savefig(OUT / "F5_station_map.pdf", **save_kw)


FIGURES = {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("figures", nargs="+", choices=sorted(FIGURES))
    ap.add_argument(
        "--dev",
        action="store_true",
        help="read the E011 dev check instead of the locked run",
    )
    ap.add_argument(
        "--paper",
        action="store_true",
        help="export clean publication figures without suptitles to paper/agu_latex/figures",
    )
    args = ap.parse_args()
    global OUT, PAPER
    PAPER = bool(args.paper)
    if args.dev:
        SOURCE.update(dir=E011 / "dev_check", tag="dev")
        OUT = (
            OUT / "dev_preview"
        )  # previews from the development check, never the paper figures
    elif args.paper:
        OUT = paths.ROOT / "paper" / "agu_latex" / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    for f in args.figures:
        FIGURES[f]()
        print(f"{f} written", file=sys.stderr)


if __name__ == "__main__":
    main()
