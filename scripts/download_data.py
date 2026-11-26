"""Download raw data into data/raw/ and record each file in data/raw/MANIFEST.csv.

Cached files are never re-downloaded or overwritten, so every command can be re-run.

    uv run scripts/download_data.py indices                  # GFZ Kp file + Kyoto Dst months
    uv run scripts/download_data.py stations                 # IMAGE station table
    uv run scripts/download_data.py image --groups training validation development candidates
    uv run scripts/download_data.py image-range --start 2017-08-01 --end 2018-09-01 --quiet-only
    uv run scripts/download_data.py chaos                    # CHAOS model files
    uv run scripts/download_data.py igrf                     # IGRF coefficient files
    uv run scripts/download_data.py verify                   # MANIFEST hashes vs files

IMAGE days per group: training and validation as configured; development and test events from
`lookback_days` before the window start to the window end; candidates likewise, with the window
placed by the Dst rule in config/defaults.yaml (event_screening). `indices` must run first.
image-range takes every UT day in [start, end), or with --quiet-only the days whose eight Kp
values are all at most --kp-max (default 2+).
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from p1 import config, events, storms
from p1.io import chaos, dst, igrf, image, kp, manifest

GROUPS = ("training", "validation", "development", "test", "candidates")


def months_between(start, end) -> list[pd.Period]:
    return list(
        pd.period_range(
            pd.Timestamp(start), pd.Timestamp(end) - pd.Timedelta(hours=1), freq="M"
        )
    )


def cmd_indices(args) -> None:
    path, new = kp.download(refresh=args.refresh)
    print(f"Kp: {path.name} ({'downloaded' if new else 'cached'})")
    cfg = events.screening_cfg()
    spans = events.dst_span_for_candidates(cfg)
    for p in events.fixed_periods("development") + events.fixed_periods("test"):
        spans.append((p["start"] - pd.Timedelta(days=1), p["end"]))
    months = sorted({m for lo, hi in spans for m in months_between(lo, hi)})
    for m in months:
        product, path, new = dst.download_month(m.year, m.month, refresh=args.refresh)
        print(f"Dst {m}: {product} {path.name} ({'downloaded' if new else 'cached'})")


def cmd_stations(args) -> None:
    path, new = image.download_station_table(refresh=args.refresh)
    print(f"IMAGE station table: {path.name} ({'downloaded' if new else 'cached'})")


def image_days(
    group: str, lookback_days: int, only: list[str] | None = None
) -> list[pd.Timestamp]:
    if group in ("training", "validation"):
        periods = events.fixed_periods(group)
        return [d for p in periods for d in events.days_between(p["start"], p["end"])]
    if group in ("development", "test"):
        periods = events.fixed_periods(group)
    elif group == "candidates":
        spans = events.dst_span_for_candidates()
        series = [dst.read(lo, hi)[0] for lo, hi in spans]
        periods = events.candidate_windows(pd.concat(series).sort_index())
    else:
        raise ValueError(group)
    if only:
        periods = [p for p in periods if p["id"] in only]
    back = pd.Timedelta(days=lookback_days)
    return [
        d for p in periods for d in events.days_between(p["start"] - back, p["end"])
    ]


def cmd_image(args) -> None:
    stations = config.station_set(args.station_set)
    lookback = args.lookback_days or events.screening_cfg()["lookback_days"]
    failed = []
    for group in args.groups:
        days = image_days(group, lookback, args.only)
        print(
            f"{group}: {len(days)} days, stations {'_'.join(sorted(stations))}",
            file=sys.stderr,
        )
        counts = image.download_days(days, stations, delay_s=args.delay)
        failed += counts["failed_days"]
        print(
            f"{group}: downloaded {counts['downloaded']}, cached {counts['cached']}, "
            f"failed {len(counts['failed_days'])}"
        )
    if failed:
        sys.exit(f"{len(failed)} day(s) failed; re-run to fetch them: {failed}")


def cmd_image_range(args) -> None:
    stations = config.station_set(args.station_set)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    days = list(pd.date_range(start, end, freq="D", inclusive="left"))
    if args.quiet_only:
        three, _ = kp.read()
        quiet = set(
            storms.quiet_days(three, end, (end - start).days, kp_max=args.kp_max)
        )
        days = [d for d in days if d in quiet]
    print(f"{len(days)} days, stations {'_'.join(sorted(stations))}", file=sys.stderr)
    counts = image.download_days(days, stations, delay_s=args.delay)
    failed = counts["failed_days"]
    print(
        f"downloaded {counts['downloaded']}, cached {counts['cached']}, failed {len(failed)}"
    )
    if failed:
        sys.exit(f"{len(failed)} day(s) failed; re-run to fetch them: {failed}")


def cmd_chaos(args) -> None:
    for version in args.versions:
        path, new = chaos.download(version)
        print(f"{version}: {path.name} ({'downloaded' if new else 'cached'})")


def cmd_igrf(args) -> None:
    for version in args.versions:
        path, new = igrf.download(version)
        print(f"{version}: {path.name} ({'downloaded' if new else 'cached'})")


def cmd_verify(args) -> None:
    problems = manifest.verify()
    rows = manifest.read_manifest()
    print(f"MANIFEST rows: {len(rows)}; problems: {len(problems)}")
    for p in problems:
        print("  " + p)
    if problems:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("indices", "stations"):
        p = sub.add_parser(name)
        p.add_argument(
            "--refresh",
            action="store_true",
            help="fetch a new dated copy of revisable products",
        )
    p = sub.add_parser("image")
    p.add_argument("--groups", nargs="+", choices=GROUPS, default=["candidates"])
    p.add_argument("--station-set", default="leave_one_out_12")
    p.add_argument(
        "--lookback-days",
        type=int,
        help="days before each event window (default: event_screening.lookback_days)",
    )
    p.add_argument("--only", nargs="+", help="event ids to restrict to")
    p.add_argument(
        "--delay", type=float, default=1.0, help="seconds to wait after each request"
    )
    p = sub.add_parser("image-range")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True, help="exclusive")
    p.add_argument("--quiet-only", action="store_true")
    p.add_argument("--kp-max", default="2+")
    p.add_argument("--station-set", default="leave_one_out_12")
    p.add_argument("--delay", type=float, default=1.0)
    p = sub.add_parser("chaos")
    p.add_argument("--versions", nargs="+", default=list(chaos.URLS))
    p = sub.add_parser("igrf")
    p.add_argument("--versions", nargs="+", default=list(igrf.URLS))
    sub.add_parser("verify")
    args = ap.parse_args()
    {
        "indices": cmd_indices,
        "stations": cmd_stations,
        "image": cmd_image,
        "image-range": cmd_image_range,
        "chaos": cmd_chaos,
        "igrf": cmd_igrf,
        "verify": cmd_verify,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
