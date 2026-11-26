"""IGRF coefficient files from NOAA NCEI (IAGA V-MOD), the causal main field (protocol section 4).

Each generation is one immutable text file, stored once as data/raw/igrf/<version>.txt. A
generation is causal for an analysis time only if it was released before that time; its field
after the last definitive epoch comes from the predictive secular variation, linearly
extrapolated by chaosmagpy. RELEASED is the end of the month in which IAGA released the
generation, so the test is conservative.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import http, manifest

SOURCE = "NOAA NCEI / IAGA V-MOD"
URLS = {
    "IGRF-12": "https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf12coeffs.txt",
    "IGRF-13": "https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf13coeffs.txt",
}
RELEASED = {"IGRF-12": "2014-12-31", "IGRF-13": "2019-12-31"}
NOTES = {
    "IGRF-12": "Thebault et al. 2015, Earth Planets Space 67:79; predictive SV 2015-2020",
    "IGRF-13": "Alken et al. 2021, Earth Planets Space 73:49; predictive SV 2020-2025",
}


def relpath(version: str) -> str:
    return f"igrf/{version}.txt"


def download(version: str, *, root=None, fetch=None) -> tuple[Path, bool]:
    """Cache one IGRF generation. Returns (path, downloaded)."""
    root = manifest.raw_root(root)
    dest = root / relpath(version)
    if dest.exists():
        return dest, False
    content, url = (fetch or http.fetch)(URLS[version])
    if b"g/h n m" not in content[:4000]:
        raise ValueError(f"{version}: not an IGRF coefficient table")
    manifest.store_raw(
        relpath(version),
        content,
        source=SOURCE,
        product="IGRF Gauss coefficients (.txt)",
        version=f"{version}, released {RELEASED[version]}",
        url=url,
        note=NOTES[version],
        root=root,
    )
    return dest, True


def causal_version(time) -> str:
    """The latest generation released before `time`."""
    t = pd.Timestamp(time)
    out = [v for v, r in RELEASED.items() if pd.Timestamp(r) < t]
    if not out:
        raise ValueError(f"no IGRF generation in URLS was released before {t}")
    return max(out, key=lambda v: pd.Timestamp(RELEASED[v]))


def load(version: str, *, root=None):
    """chaosmagpy model of a cached generation."""
    import chaosmagpy as cp

    path = manifest.raw_root(root) / relpath(version)
    if not path.exists():
        raise FileNotFoundError(
            f"{version} not cached; run scripts/download_data.py igrf"
        )
    return cp.load_IGRF_txtfile(str(path), name=version)
