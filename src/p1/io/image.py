"""IMAGE magnetometer data (FMI): download one file per UT day, parse, and read the station table.

Request: data_download.php with format=text, sample_rate=60, length=1440 (minutes). The 1-min values
are the mean of the six 10-s samples in [t, t+60 s), stamped at t (checked against 10-s data for
KEV on 2017-08-01). Missing values are 99999.9. Components are geographic X (north), Y (east) and
Z (down), in nT. The server returns stations in its own column order; parsing goes by the header.

read_range is the only reader of cached data and refuses locked test events and their reference
periods (p1.io.locked; CLAUDE.md hard rule 1).
"""

from __future__ import annotations

import datetime as dt
import html
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import http, locked, manifest

DATA_URL = "https://space.fmi.fi/image/www/data_download.php"
STATION_TABLE_URL = "https://space.fmi.fi/image/www/index.php?page=stations"
SOURCE = "FMI IMAGE"
PRODUCT = "image_text_60s"
MISSING_FROM = 99999.0  # values >= this are missing (IMAGE writes 99999.9)
COMPONENTS = ("X", "Y", "Z")
MINUTES_PER_DAY = 1440


def _date(day) -> dt.date:
    return pd.Timestamp(day).date()


def day_relpath(day, stations) -> str:
    d = _date(day)
    return f"image/text_60s/{d:%Y}/image_{d:%Y%m%d}_{'_'.join(sorted(stations))}.txt"


def request_params(day, stations) -> dict:
    d = _date(day)
    return {
        "starttime": f"{d:%Y%m%d}00",
        "length": MINUTES_PER_DAY,
        "format": "text",
        "sample_rate": 60,
        "stations": "_".join(sorted(stations)),
    }


def parse_text(
    content: str,
) -> tuple[list[tuple[str, str]], pd.DatetimeIndex, np.ndarray]:
    """Parse an IMAGE `text` file into (columns, times, values); missing values become NaN."""
    lines = content.splitlines()
    if len(lines) < 2:
        raise ValueError("IMAGE file has no data lines")
    head = lines[0].split()
    if head[:6] != ["YYYY", "MM", "DD", "HH", "MM", "SS"] or len(head) % 2:
        raise ValueError(f"unexpected IMAGE header: {lines[0][:80]!r}")
    if lines[1].strip().strip("-"):
        raise ValueError("IMAGE header separator line missing")
    columns = list(zip(head[6::2], head[7::2]))
    rows = [line.split() for line in lines[2:] if line.strip()]
    if not rows or any(len(r) != 6 + len(columns) for r in rows):
        raise ValueError("IMAGE data rows do not match the header")
    a = np.array(rows, dtype=float)
    t = a[:, :6].astype(int)
    times = pd.DatetimeIndex(
        pd.to_datetime(
            {
                "year": t[:, 0],
                "month": t[:, 1],
                "day": t[:, 2],
                "hour": t[:, 3],
                "minute": t[:, 4],
                "second": t[:, 5],
            }
        ),
        name="time",
    )
    values = a[:, 6:]
    values[values >= MISSING_FROM] = np.nan
    return columns, times, values


def validate_day(content: str, day) -> list[tuple[str, str]]:
    """Check a downloaded day: header, 1440 rows, minutes 00:00..23:59 of that UT day. Returns columns."""
    columns, times, _ = parse_text(content)
    expected = pd.date_range(
        pd.Timestamp(_date(day)), periods=MINUTES_PER_DAY, freq="min"
    )
    if len(times) != MINUTES_PER_DAY or not (times == expected).all():
        raise ValueError(
            f"IMAGE file for {_date(day)} does not cover 00:00-23:59 at 1-min spacing"
        )
    return columns


def download_day(day, stations, *, root=None, fetch=None) -> tuple[Path, bool]:
    """Download one UT day for the station set unless cached. Returns (path, downloaded)."""
    root = manifest.raw_root(root)
    rel = day_relpath(day, stations)
    dest = root / rel
    if dest.exists():
        return dest, False
    params = request_params(day, stations)
    content, url = (fetch or http.fetch)(DATA_URL, params)
    columns = validate_day(content.decode("latin-1"), day)
    absent = sorted(set(stations) - {s for s, _ in columns})
    manifest.store_raw(
        rel,
        content,
        source=SOURCE,
        product=PRODUCT,
        version="server-current (no version label)",
        url=url,
        params=params,
        note=f"absent from file: {' '.join(absent)}" if absent else "",
        root=root,
    )
    return dest, True


def download_days(
    days, stations, *, root=None, fetch=None, delay_s: float = 1.0, log=sys.stderr
) -> dict:
    """Download several UT days, pausing delay_s after each request.

    A day that fails (network error or invalid file) is skipped and listed in counts["failed_days"];
    nothing is stored for it, so a re-run fetches only the missing days.
    """
    days = sorted({_date(d) for d in days})
    counts = {"downloaded": 0, "cached": 0, "failed_days": []}
    for i, d in enumerate(days, 1):
        try:
            _, new = download_day(d, stations, root=root, fetch=fetch)
            status = "downloaded" if new else "cached"
            counts[status] += 1
        except (OSError, ValueError) as e:  # requests errors are OSError subclasses
            new, status = True, f"FAILED ({type(e).__name__}: {str(e)[:120]})"
            counts["failed_days"].append(d)
        if log:
            print(f"[image {i}/{len(days)}] {d} {status}", file=log, flush=True)
        if new and delay_s:
            time.sleep(delay_s)
    return counts


def _read_day(path: Path, *, validity_only: bool = False) -> pd.DataFrame:
    """One day file as a DataFrame indexed by minute, columns (station, component).

    Internal: callers go through read_range, which applies the locked-event guard.
    """
    columns, times, values = parse_text(Path(path).read_text(encoding="latin-1"))
    cols = pd.MultiIndex.from_tuples(columns, names=["station", "component"])
    if validity_only:
        return pd.DataFrame(~np.isnan(values), index=times, columns=cols)
    return pd.DataFrame(values, index=times, columns=cols)


def read_range(
    start, end, stations, *, validity_only: bool = False, root=None
) -> pd.DataFrame:
    """Minutes in [start, end) for the station set, from cached day files.

    validity_only=True returns booleans (value present) and discards the values. Stations absent
    from a file come back as NaN (or False). Raises LockedEventError if [start, end) touches a
    locked test event or its reference period (also for validity_only), and FileNotFoundError
    for an uncached day.
    """
    locked.check(start, end)
    root = manifest.raw_root(root)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = pd.date_range(
        start.normalize(), (end - pd.Timedelta(minutes=1)).normalize(), freq="D"
    )
    want = pd.MultiIndex.from_product(
        [sorted(stations), COMPONENTS], names=["station", "component"]
    )
    frames = []
    for d in days:
        path = root / day_relpath(d, stations)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not cached; run scripts/download_data.py image"
            )
        df = _read_day(path, validity_only=validity_only)
        frames.append(
            df.reindex(columns=want, fill_value=False)
            if validity_only
            else df.reindex(columns=want)
        )
    out = pd.concat(frames)
    return out.loc[(out.index >= start) & (out.index < end)]


# --- station table -------------------------------------------------------------------------------

STATION_TABLE_GLOB = "image/stations/image_stations_dl*.html"


def download_station_table(
    *, refresh: bool = False, root=None, fetch=None, today=None
) -> tuple[Path, bool]:
    """Cache the IMAGE station table page (dated file name; reused unless refresh)."""
    root = manifest.raw_root(root)
    cached = manifest.latest_dated(root, STATION_TABLE_GLOB)
    if cached and not refresh:
        return cached, False
    d = today or manifest.today_utc()
    rel = f"image/stations/image_stations_dl{d:%Y%m%d}.html"
    if (root / rel).exists():
        return root / rel, False
    content, url = (fetch or http.fetch)(STATION_TABLE_URL)
    table = parse_station_table(content.decode("utf-8"))
    if len(table) < 10:
        raise ValueError("IMAGE station table looks truncated")
    manifest.store_raw(
        rel,
        content,
        source=SOURCE,
        product="image_station_table",
        version=f"page as of {d:%Y-%m-%d}",
        url=url,
        note="CGM coordinates for epoch 2001, altitude 0 km (page footnote)",
        root=root,
    )
    return root / rel, True


def parse_station_table(page: str) -> pd.DataFrame:
    """Station table: code, name, lat, lon (east, -180..180), cgm_lat, cgm_lon, provider, network, start, end."""
    m = re.search(r"(?is)<table.*?</table>", page)
    if not m:
        raise ValueError("no table in IMAGE station page")
    records = []
    for row in re.findall(r"(?is)<tr.*?</tr>", m.group(0)):
        cells = [
            html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
            for c in re.findall(r"(?is)<t[hd][^>]*>(.*?)</t[hd]>", row)
        ]
        if len(cells) == 11 and cells[0].isdigit():
            records.append(cells[1:])
    df = pd.DataFrame(
        records,
        columns=[
            "code",
            "name",
            "glat",
            "glon",
            "cgm_lat",
            "cgm_lon",
            "provider",
            "network",
            "start",
            "end",
        ],
    )
    for c in ["glat", "glon", "cgm_lat", "cgm_lon"]:
        df[c] = df[c].astype(float)
    df["lat"] = df["glat"]
    df["lon"] = (df["glon"] + 180.0) % 360.0 - 180.0
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")  # empty = still operating
    return df.set_index("code")


def read_station_table(path: Path | None = None, *, root=None) -> pd.DataFrame:
    path = path or manifest.latest_dated(manifest.raw_root(root), STATION_TABLE_GLOB)
    if path is None:
        raise FileNotFoundError(
            "IMAGE station table not cached; run scripts/download_data.py stations"
        )
    return parse_station_table(Path(path).read_text(encoding="utf-8"))
