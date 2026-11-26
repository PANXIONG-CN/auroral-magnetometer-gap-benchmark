"""CHAOS geomagnetic field model files from DTU Space (docs/02_data_sources.md; D03).

Each version is one immutable MATLAB file, stored once as data/raw/chaos/<version>.mat. RELEASED
holds the Last-Modified date of the file on the DTU server; a model is causal for an analysis time
only if it was released before that time (protocol section 4, hard rule 3). Results that use a
later release are labelled offline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import http, manifest

SOURCE = "DTU Space"
URLS = {
    "CHAOS-8.6": "https://www.spacecenter.dk/files/magnetic-models/CHAOS-8/_downloads/4573c39d571e013a8374f83397759710/CHAOS-8.6.mat",
    "CHAOS-6-x6": "https://www.spacecenter.dk/files/magnetic-models/CHAOS-6/CHAOS-6-x6.mat",
}
RELEASED = {"CHAOS-8.6": "2026-06-12", "CHAOS-6-x6": "2018-06-22"}
NOTES = {
    "CHAOS-8.6": "latest release; Kloss et al. 2026 (doi:10.1186/s40623-025-02352-0)",
    "CHAOS-6-x6": "Swarm data to end of April 2018, ground observatories to April 2018",
}


def relpath(version: str) -> str:
    return f"chaos/{version}.mat"


def download(version: str, *, root=None, fetch=None) -> tuple[Path, bool]:
    """Cache one CHAOS version. Returns (path, downloaded)."""
    root = manifest.raw_root(root)
    dest = root / relpath(version)
    if dest.exists():
        return dest, False
    content, url = (fetch or http.fetch)(URLS[version])
    if not content.startswith(b"MATLAB"):
        raise ValueError(f"{version}: not a MATLAB file")
    manifest.store_raw(
        relpath(version),
        content,
        source=SOURCE,
        product="CHAOS model coefficients (.mat)",
        version=f"{version}, released {RELEASED[version]}",
        url=url,
        note=NOTES[version],
        root=root,
    )
    return dest, True


def is_causal(version: str, time) -> bool:
    """True if the version was released before `time`."""
    return pd.Timestamp(RELEASED[version]) < pd.Timestamp(time)


def load(version: str, *, root=None):
    """chaosmagpy CHAOS model of a cached version."""
    import chaosmagpy as cp

    path = manifest.raw_root(root) / relpath(version)
    if not path.exists():
        raise FileNotFoundError(
            f"{version} not cached; run scripts/download_data.py chaos"
        )
    return cp.load_CHAOS_matfile(str(path))
