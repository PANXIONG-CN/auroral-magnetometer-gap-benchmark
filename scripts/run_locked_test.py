"""E011: the one-time evaluation on the locked test events (task T10).

    uv run scripts/run_locked_test.py --pi-signoff "<PI name, date: wording>"
    uv run scripts/run_locked_test.py --dev        # end-to-end check on the development events

This is the only program that may read locked test events (CLAUDE.md hard rule 1): it opens them
with p1.io.locked.locked_access(), which refuses any other caller. It runs once, after the PI signs
off, from a committed state; every result carries the git commit hash and the sign-off. A run log
is written before the first locked byte is read; while it exists the script refuses to run again
(a failed run needs the PI's decision before the log is moved away).

Methods and rules are frozen in experiments/E011_locked_test/config.yaml and p1.final_eval. --dev
runs the same code on the development events and writes to experiments/E011_locked_test/dev_check.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from p1 import events, final_eval, paths, provenance
from p1.io import locked

EXP = paths.ROOT / "experiments" / "E011_locked_test"


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write(ana: dict, res: dict, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for k, v in {**res, **ana}.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(out / f"{k}_{stem}.csv", index=k == "bias", float_format="%.4f")
    g = {k: v for k, v in ana["c2_gate_a_refit"].items() if k != "curve"}
    (out / f"c2_gate_a_refit_{stem}.json").write_text(
        json.dumps(g, indent=2, default=float)
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pi-signoff", help="PI sign-off (name, date), stored in results")
    ap.add_argument(
        "--dev", action="store_true", help="run on the development events instead"
    )
    args = ap.parse_args()
    if not args.dev and not args.pi_signoff:
        sys.exit("the locked run needs --pi-signoff")

    cfg = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    for key in ("d_star_km",):
        if cfg[key] is None:
            sys.exit(f"config {key} is not frozen yet")
    if cfg["waterfall"]["temporal_method"] is None:
        sys.exit("config waterfall.temporal_method is not frozen yet")
    git = provenance.git_short_hash()
    if git.endswith("-dirty"):
        sys.exit(
            "commit src/, scripts/, config/ and experiments/ first: results must carry a clean hash"
        )
    full = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=paths.ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    cfg_hash = provenance.files_hash(
        EXP / "config.yaml",
        *(paths.CONFIG / n for n in ("defaults.yaml", "events.yaml", "stations.yaml")),
    )

    if args.dev:
        out, tag = EXP / "dev_check", "dev"
        periods = events.fixed_periods("development")
        res = final_eval.collect(
            cfg, periods, git=git, cfg_hash=cfg_hash, log=lambda m: print(m, flush=True)
        )
    else:
        out, tag = EXP / "results", "locked"
        out.mkdir(parents=True, exist_ok=True)
        logs = sorted(out.glob("run_log_*.json"))
        if logs:
            sys.exit(f"E011 has already been run ({logs[0].name}); it runs only once")
        log = {
            "experiment": "E011_locked_test",
            "git": git,
            "git_full": full,
            "cfg": cfg_hash,
            "pi_signoff": args.pi_signoff,
            "started_utc": now(),
            "status": "started",
        }
        log_path = out / f"run_log_{git}_cfg{cfg_hash}.json"
        log_path.write_text(
            json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with locked.locked_access():
            periods = events.fixed_periods("test")
            res = final_eval.collect(
                cfg,
                periods,
                git=git,
                cfg_hash=cfg_hash,
                log=lambda m: print(m, flush=True),
            )
    ana = final_eval.analyse(res, cfg)
    stem = f"{tag}_{git}_cfg{cfg_hash}"
    write(ana, res, out, stem)
    verdicts = {
        "c1": ana["c1_verdict"],
        "c2": ana["c2_verdict"],
        "c3": ana["c3_verdict"],
    }
    if not args.dev:
        log.update(
            finished_utc=now(),
            status="finished",
            events=[p["id"] for p in periods],
            verdicts=verdicts,
        )
        log_path.write_text(
            json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    pd.set_option("display.width", 220)
    print(json.dumps(verdicts, indent=1))
    print(ana["c1_paired"].round(2).to_string(index=False))
    print(ana["c1_flagged"].to_string(index=False))
    print(ana["c1_clean_paired"].round(2).to_string(index=False))
    print(ana["c2_tests"].round(3).to_string(index=False))
    print(
        ana["c3_gains"][
            ["kp_group", "station_set", "comparison", "mean", "lo", "hi", "n_positive"]
        ]
        .round(2)
        .to_string(index=False)
    )
    print(
        ana["waterfall_summary"][
            ["station_set", "step", "rmse_h", "rmse_h_lo", "rmse_h_hi"]
        ]
        .round(2)
        .to_string(index=False)
    )
    print(f"written: {out.relative_to(paths.ROOT)}/*_{stem}.*")


if __name__ == "__main__":
    main()
