"""Hourly Dst from the WDC for Geomagnetism, Kyoto (monthly HTML pages).

Used only to verify storm intensity and main-phase timing of candidate events. A month is taken
from the best product available: final > provisional > realtime. Final pages do not change and
are stored once; provisional and realtime pages carry the download date in the file name.
Column h (1..24) of a page is the mean over [h-1, h) UT; values are stamped at the hour start.
"""

from __future__ import annotations

import calendar
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import http, manifest

BASE = "https://wdc.kugi.kyoto-u.ac.jp"
SOURCE = "WDC for Geomagnetism, Kyoto"
PRODUCTS = ("final", "provisional", "realtime")
_DAY_LINE = re.compile(r"^([ \d]\d) ")


def month_url(product: str, year: int, month: int) -> str:
    return f"{BASE}/dst_{product}/{year:04d}{month:02d}/index.html"


def _relpath(product: str, year: int, month: int, dl=None) -> str:
    if product == "final":
        return f"dst/final/dst_{year:04d}{month:02d}.html"
    return f"dst/{product}/dst_{year:04d}{month:02d}_dl{dl:%Y%m%d}.html"


def cached(year: int, month: int, *, root=None) -> tuple[str, Path] | None:
    """Best cached (product, path) for a month, or None."""
    root = manifest.raw_root(root)
    final = root / _relpath("final", year, month)
    if final.exists():
        return "final", final
    for product in PRODUCTS[1:]:
        p = manifest.latest_dated(
            root, f"dst/{product}/dst_{year:04d}{month:02d}_dl*.html"
        )
        if p:
            return product, p
    return None


def download_month(
    year: int, month: int, *, refresh: bool = False, root=None, fetch=None, today=None
):
    """Cache one month. Returns (product, path, downloaded)."""
    root = manifest.raw_root(root)
    have = cached(year, month, root=root)
    if have and (have[0] == "final" or not refresh):
        return have[0], have[1], False
    fetch = fetch or http.fetch
    d = today or manifest.today_utc()
    for product in PRODUCTS:
        rel = _relpath(product, year, month, d)
        if (root / rel).exists():
            return product, root / rel, False
        try:
            content, url = fetch(month_url(product, year, month))
        except http.NotFound:
            continue
        values = parse_month(content.decode("latin-1"), year, month)
        if values.notna().sum() == 0:
            continue
        manifest.store_raw(
            rel,
            content,
            source=SOURCE,
            product=f"dst_{product}",
            version=product if product == "final" else f"{product} as of {d:%Y-%m-%d}",
            url=url,
            root=root,
        )
        return product, root / rel, True
    raise FileNotFoundError(f"no Dst product found for {year:04d}-{month:02d}")


def parse_month(page: str, year: int, month: int) -> pd.Series:
    """Hourly Dst (nT) for one month page; blank or 9999 fields become NaN.

    Fields are fixed width (4 characters), so adjacent negative values such as '-33-131' are split
    by position: day in columns 0-1, then three groups of eight fields starting at 3, 36 and 69.
    """
    m = re.search(r'(?is)<pre[^>]*class="data"[^>]*>(.*?)</pre>', page)
    block = m.group(1) if m else page
    ndays = calendar.monthrange(year, month)[1]
    out = np.full((ndays, 24), np.nan)
    for line in block.splitlines():
        hit = _DAY_LINE.match(line)
        if not hit:
            continue
        day = int(hit.group(1))
        if not 1 <= day <= ndays:
            raise ValueError(f"bad day {day} in Dst page {year}-{month:02d}")
        line = line.ljust(101)
        for h in range(24):
            start = 3 + 33 * (h // 8) + 4 * (h % 8)
            field = line[start : start + 4].strip()
            if field and field != "9999":
                out[day - 1, h] = int(field)
    index = pd.date_range(
        pd.Timestamp(year, month, 1), periods=ndays * 24, freq="h", name="time"
    )
    return pd.Series(out.ravel(), index=index, name="dst")


def read(start, end, *, root=None) -> tuple[pd.Series, dict]:
    """Hourly Dst in [start, end) from cached months, and the product used for each month."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    months = pd.period_range(start, end - pd.Timedelta(hours=1), freq="M")
    parts, used = [], {}
    for p in months:
        have = cached(p.year, p.month, root=root)
        if have is None:
            raise FileNotFoundError(
                f"Dst {p} not cached; run scripts/download_data.py indices"
            )
        used[str(p)] = have[0]
        parts.append(
            parse_month(have[1].read_text(encoding="latin-1"), p.year, p.month)
        )
    s = pd.concat(parts)
    return s[(s.index >= start) & (s.index < end)], used
