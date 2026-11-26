"""Kp, ap and Ap from GFZ Potsdam (file Kp_ap_Ap_SN_F107_since_1932.txt, CC BY 4.0).

Cite Matzka et al. (2021), Space Weather, doi:10.1029/2020SW002641, and the data publication
doi:10.5880/Kp.0001. Kp is used only to select quiet days and to group results by activity
(docs/02_data_sources.md); it is never a model input. Each 3-hour value is stamped at the start
of its interval (Kp1 = 00-03 UT).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import http, manifest

URL = "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
SOURCE = "GFZ Potsdam"
PRODUCT = "Kp_ap_Ap_SN_F107_since_1932"
GLOB = f"kp/{PRODUCT}_dl*.txt"
_SUFFIX = {"-": -1, "o": 0, "": 0, "+": 1}


def kp_thirds(value) -> int:
    """Kp in thirds: '2+' -> 7, '5-' -> 14, '5o' or 5 -> 15, 2.333 -> 7."""
    if isinstance(value, str):
        v = value.strip()
        return 3 * int(v[0]) + _SUFFIX[v[1:]]
    return round(float(value) * 3)


def download(
    *, refresh: bool = False, root=None, fetch=None, today=None
) -> tuple[Path, bool]:
    """Cache the GFZ file (dated file name; the cached copy is reused unless refresh)."""
    root = manifest.raw_root(root)
    cached = manifest.latest_dated(root, GLOB)
    if cached and not refresh:
        return cached, False
    d = today or manifest.today_utc()
    rel = f"kp/{PRODUCT}_dl{d:%Y%m%d}.txt"
    if (root / rel).exists():
        return root / rel, False
    content, url = (fetch or http.fetch)(URL)
    three_hourly, _ = parse(content.decode("utf-8"))
    if three_hourly.empty:
        raise ValueError("GFZ Kp file has no data rows")
    manifest.store_raw(
        rel,
        content,
        source=SOURCE,
        product=PRODUCT,
        version=f"file as of {d:%Y-%m-%d}; per-day definitive flag D in column 28",
        url=url,
        note="CC BY 4.0; cite Matzka et al. 2021 (doi:10.1029/2020SW002641) and doi:10.5880/Kp.0001",
        root=root,
    )
    return root / rel, True


def parse(text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(three_hourly, daily). three_hourly: kp, ap, definitive, indexed by interval start.
    daily: Ap, D, kp_max, kp_sum, indexed by UT date. Missing Kp (-1) becomes NaN."""
    rows = [
        line.split()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    a = np.array([r for r in rows if len(r) == 28], dtype=float)
    if a.size == 0:
        return pd.DataFrame(columns=["kp", "ap", "definitive"]), pd.DataFrame()
    days = pd.to_datetime(
        {
            "year": a[:, 0].astype(int),
            "month": a[:, 1].astype(int),
            "day": a[:, 2].astype(int),
        }
    )
    kp = a[:, 7:15]
    ap = a[:, 15:23]
    kp = np.where(kp < 0, np.nan, kp)
    ap = np.where(ap < 0, np.nan, ap)
    flag = a[:, 27].astype(int)
    times = (
        days.values[:, None] + (np.arange(8) * np.timedelta64(3, "h"))[None, :]
    ).ravel()
    three_hourly = pd.DataFrame(
        {"kp": kp.ravel(), "ap": ap.ravel(), "definitive": np.repeat(flag >= 1, 8)},
        index=pd.DatetimeIndex(times, name="time"),
    )
    daily = pd.DataFrame(
        {
            "Ap": a[:, 23],
            "D": flag,
            "kp_max": np.nanmax(kp, axis=1),
            "kp_sum": np.nansum(kp, axis=1),
        },
        index=pd.DatetimeIndex(days, name="date"),
    )
    return three_hourly, daily


def read(path: Path | None = None, *, root=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = path or manifest.latest_dated(manifest.raw_root(root), GLOB)
    if path is None:
        raise FileNotFoundError(
            "Kp file not cached; run scripts/download_data.py indices"
        )
    return parse(Path(path).read_text(encoding="utf-8"))
