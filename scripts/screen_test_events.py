"""T1 test-event screening (docs/03_evaluation_protocol.md section 2; rules in docs/05_decisions.md D14).

    uv run scripts/screen_test_events.py

For each candidate test storm: Dst minimum and main-phase timing, Kp in the window, quiet days
before the window, and IMAGE completeness of the 12 leave-one-out stations. Development events get
the same table, so the station rule can be compared with the student's E004 counts.

IMAGE files are read as availability masks only (image.read_range(validity_only=True)); magnetometer
values of candidate events are never loaded. The script refuses to run once test events are locked
(CLAUDE.md hard rule 1).

Writes results/t1_event_screening/{events,coverage,tables}_<git>_cfg<hash>.{csv,md}.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from p1 import config, events, paths, provenance, storms
from p1.io import dst, image, kp

OUT_DIR = paths.RESULTS / "t1_event_screening"
CONFIG_FILES = ("events.yaml", "stations.yaml", "defaults.yaml")
DAY = pd.Timedelta(days=1)

# Rule sensitivity, kept as the record behind the PI's choice (D13 -> D14). Quiet-day variants:
# (lookback days, highest Kp allowed). Station variants: W90 = at least 9 stations that each
# have a reference and >= 90 % window coverage (D13); M99 = the minute-by-minute rule (D14).
SENS_QUIET = {"10d<=2+": (10, "2+"), "20d<=2+": (20, "2+"), "10d<=3-": (10, "3-")}
W90_WINDOW_COVERAGE = 0.9
ADOPTED = ("20d<=2+", "M99")
ORIGINAL = ("10d<=2+", "W90")


def event_list(cfg: dict) -> list[dict]:
    """Candidates with Dst-placed windows, then development events with configured windows."""
    spans = events.dst_span_for_candidates(cfg)
    dev = events.fixed_periods("development")
    spans += [(p["start"] - DAY, p["end"]) for p in dev]
    series = pd.concat([dst.read(lo, hi)[0] for lo, hi in spans]).sort_index()
    series = series[~series.index.duplicated()]
    out = [dict(w, role="candidate") for w in events.candidate_windows(series, cfg)]
    for p in dev:
        mp = storms.main_phase(
            series,
            p["start"] + DAY,
            search_days=tuple(cfg["dst_search_days"]),
            onset_lookback_h=cfg["onset_lookback_h"],
        )
        out.append(dict(p, **mp, role="development", storm_date=pd.NaT))
    return out


def sensitivity(ev, cfg, kp3h, stations, valid) -> dict:
    """Pass/fail under each quiet-day variant x station rule."""
    start, end = ev["start"], ev["end"]
    need = cfg["min_valid_stations"]
    window_cov = storms.coverage(valid, start, end, stations)
    out = {}
    for name, (lookback, kp_max) in SENS_QUIET.items():
        days = storms.quiet_days(kp3h, start, lookback, kp_max=kp_max)
        cov = storms.quiet_day_coverage(valid, days, stations)
        has_ref = cov >= cfg["quiet_day_min_coverage"]
        ref = [s for s in stations if has_ref[s]]
        w90 = int(((window_cov >= W90_WINDOW_COVERAGE) & has_ref).sum()) >= need
        m99 = storms.per_minute_rule(
            valid, start, end, ref, min_valid=need, min_frac=cfg["min_frac_minutes"]
        )["ok"]
        out[f"pass_{name}_W90"] = bool(days) and w90
        out[f"pass_{name}_M99"] = bool(days) and m99
    return out


def screen(ev, cfg, kp3h, stations) -> tuple[dict, pd.DataFrame]:
    start, end = ev["start"], ev["end"]
    back = start - cfg["lookback_days"] * DAY
    _, dst_used = dst.read(ev["t_dst_min"], ev["t_dst_min"] + pd.Timedelta(hours=1))
    valid = image.read_range(back, end, stations, validity_only=True)

    quiet = storms.quiet_days(
        kp3h, start, cfg["lookback_days"], kp_max=cfg["quiet_kp_max"]
    )
    quiet_cov = storms.quiet_day_coverage(valid, quiet, stations)
    has_ref = quiet_cov >= cfg["quiet_day_min_coverage"]
    ref = [s for s in stations if has_ref[s]]
    rule = storms.per_minute_rule(
        valid,
        start,
        end,
        ref,
        min_valid=cfg["min_valid_stations"],
        min_frac=cfg["min_frac_minutes"],
    )
    window_cov = storms.coverage(valid, start, end, stations)
    lookback_cov = storms.coverage(valid, back, start, stations)

    cov = pd.DataFrame(
        {
            "event": ev["id"],
            "station": stations,
            "window_cov": window_cov.values,
            "best_quiet_cov": quiet_cov.values,
            "lookback_cov": lookback_cov.values,
            "has_reference": has_ref.values,
        }
    )
    low = [s for s in stations if window_cov[s] < W90_WINDOW_COVERAGE]
    row = {
        "event": ev["id"],
        "role": ev["role"],
        "storm_date": ev["storm_date"],
        "dst_product": "/".join(sorted(set(dst_used.values()))),
        "dst_min": ev["dst_min"],
        "t_dst_min": ev["t_dst_min"],
        "t_onset": ev["t_onset"],
        "dst_at_onset": ev["dst_at_onset"],
        "main_phase_h": ev["main_phase_h"],
        "window_start": start,
        "window_end": end,
        **storms.kp_summary(
            kp3h, start, end, active_kp=config.load("defaults")["active_threshold_kp"]
        ),
        "n_quiet_days": len(quiet),
        "quiet_days": " ".join(f"{d:%m-%d}" for d in quiet),
        "n_with_reference": len(ref),
        "no_reference": " ".join(s for s in stations if not has_ref[s]),
        "window_cov_below_90": "；".join(f"{s} {window_cov[s]:.1%}" for s in low),
        "min_valid_per_minute": rule["min_valid_per_minute"],
        "frac_minutes_ok": rule["frac_minutes_ok"],
        "passes": bool(quiet) and rule["ok"],
        **sensitivity(ev, cfg, kp3h, stations, valid),
    }
    return row, cov


def markdown(evt: pd.DataFrame, cov: pd.DataFrame, stations, cfg) -> str:
    def ts(t):
        return f"{t:%m-%d %H}" if pd.notna(t) else ""

    lines = ["### 磁暴核实（Dst、Kp）", ""]
    lines += [
        "| 事件 | 角色 | Dst 产品 | Dst 最小 (nT) | 最小时刻 | 主相起点 | 主相 (h) | Kp 最大 | Kp ≥ 5 (h) | Kp 定值 | 窗口（含首尾日） |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for _, r in evt.iterrows():
        lines.append(
            f"| {r.event} | {'候选' if r.role == 'candidate' else '开发'} | {r.dst_product} | {r.dst_min:.0f} "
            f"| {ts(r.t_dst_min)} | {ts(r.t_onset)} | {r.main_phase_h} | {r.kp_max:.2f} | {r.hours_kp_active} "
            f"| {'是' if r.kp_definitive else '否'} | {r.window_start:%Y-%m-%d} – {r.window_end - DAY:%m-%d} |"
        )
    lb = cfg["lookback_days"]
    lines += ["", f"### 完整率（回看 {lb} 天，逐分钟判定）", ""]
    lines += [
        f"| 事件 | 窗口前 {lb} 天内安静日 | 有参考的站 / 12 | 每分钟有效站最少 | ≥ {cfg['min_valid_stations']} 站有效的分钟 | 窗口覆盖 < 90% 的站 | 通过 |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for _, r in evt.iterrows():
        q = (
            f"{r.n_quiet_days}（最近 {r.quiet_days.split()[-1]}）"
            if r.n_quiet_days
            else "0"
        )
        refs = f"{r.n_with_reference}" + (
            f"（缺 {r.no_reference}）" if r.no_reference else ""
        )
        lines.append(
            f"| {r.event} | {q} | {refs} | {r.min_valid_per_minute} | {r.frac_minutes_ok:.2%} "
            f"| {r.window_cov_below_90 or '—'} | {'是' if r.passes else '否'} |"
        )

    cand = evt[evt.role == "candidate"]
    keys = [(q, s) for s in ("W90", "M99") for q in SENS_QUIET]

    def label(q, s):
        tag = (
            "（D13）" if (q, s) == ORIGINAL else "（采用）" if (q, s) == ADOPTED else ""
        )
        return f"{q} · {s}{tag}"

    lines += ["", "### 规则敏感性：候选事件在各规则组合下是否通过", ""]
    lines += [
        "| 事件 | " + " | ".join(label(q, s) for q, s in keys) + " |",
        "| --- |" + " :-: |" * len(keys),
    ]
    for _, r in cand.iterrows():
        cells = ["✓" if r[f"pass_{q}_{s}"] else "✗" for q, s in keys]
        lines.append(f"| {r.event} | " + " | ".join(cells) + " |")
    totals = [str(int(cand[f"pass_{q}_{s}"].sum())) for q, s in keys]
    lines.append("| 通过场数 | " + " | ".join(totals) + " |")

    lines += [
        "",
        f"### 窗口内各站 X、Y 同时有效的分钟比例（%；* 表示回看 {lb} 天内没有覆盖 ≥ {cfg['quiet_day_min_coverage']:.0%} 的安静日）",
        "",
    ]
    lines += [
        "| 事件 | " + " | ".join(stations) + " |",
        "| --- |" + " ---: |" * len(stations),
    ]
    for e, g in cov.groupby("event", sort=False):
        g = g.set_index("station")
        cells = [
            f"{100 * g.loc[s, 'window_cov']:.1f}"
            + ("" if g.loc[s, "has_reference"] else "*")
            for s in stations
        ]
        lines.append(f"| {e} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--no-write", action="store_true", help="print tables only")
    args = ap.parse_args()

    locked = [
        e["id"] for e in config.load("events").get("test") or [] if e.get("locked")
    ]
    if locked:
        sys.exit(
            f"test events are locked ({', '.join(map(str, locked))}); this screening reads "
            "their data and is closed (CLAUDE.md hard rule 1)"
        )

    cfg = events.screening_cfg()
    stations = config.station_set("leave_one_out_12")
    kp3h, _ = kp.read()
    rows, covs = [], []
    for ev in event_list(cfg):
        row, cov = screen(ev, cfg, kp3h, stations)
        rows.append(row)
        covs.append(cov)
    evt = pd.DataFrame(rows)
    cov = pd.concat(covs, ignore_index=True)
    md = markdown(evt, cov, stations, cfg)
    print(md)
    if not args.no_write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = provenance.result_stem("{}", *CONFIG_FILES)
        evt.to_csv(OUT_DIR / f"{stem.format('events')}.csv", index=False)
        cov.to_csv(
            OUT_DIR / f"{stem.format('coverage')}.csv", index=False, float_format="%.4f"
        )
        (OUT_DIR / f"{stem.format('tables')}.md").write_text(md, encoding="utf-8")
        print(
            f"written to {OUT_DIR.relative_to(paths.ROOT)}/ with stem {stem.format('*')}"
        )


if __name__ == "__main__":
    main()
