"""Write-once storage under data/raw/ and its MANIFEST.csv.

Rules (CLAUDE.md hard rule 6): a raw file is written once, never overwritten or deleted, and set
read-only. Every stored file gets exactly one MANIFEST row with source, product, version, request,
download time and SHA-256. Products that the provider revises (Kp, provisional Dst, the station
table) carry the download date in their file name, so a refresh adds a new file.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

from .. import paths

MANIFEST_NAME = "MANIFEST.csv"
COLUMNS = [
    "path",
    "source",
    "product",
    "version",
    "url",
    "params",
    "downloaded_utc",
    "bytes",
    "sha256",
    "note",
]


def raw_root(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else paths.DATA_RAW


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(root: Path | str | None = None) -> list[dict]:
    path = raw_root(root) / MANIFEST_NAME
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append_row(row: dict, root: Path) -> None:
    path = root / MANIFEST_NAME
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def store_raw(
    relpath: str,
    content: bytes,
    *,
    source: str,
    product: str,
    version: str,
    url: str,
    params: dict | None = None,
    note: str = "",
    root: Path | str | None = None,
) -> Path:
    """Write content to data/raw/<relpath> once and add its MANIFEST row.

    Raises FileExistsError if the file exists or the path is already in the MANIFEST.
    """
    root = raw_root(root)
    dest = root / relpath
    if dest.exists() or any(r["path"] == relpath for r in read_manifest(root)):
        raise FileExistsError(f"raw file already stored: {relpath}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(content)
    os.replace(part, dest)
    try:
        dest.chmod(0o444)
    except OSError:
        pass
    _append_row(
        {
            "path": relpath,
            "source": source,
            "product": product,
            "version": version,
            "url": url,
            "params": json.dumps(params or {}, sort_keys=True, ensure_ascii=False),
            "downloaded_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "bytes": len(content),
            "sha256": sha256_bytes(content),
            "note": note,
        },
        root,
    )
    return dest


def verify(root: Path | str | None = None) -> list[str]:
    """Problems found: hash or size mismatches, listed files that are missing, unlisted raw files."""
    root = raw_root(root)
    rows = read_manifest(root)
    problems = []
    listed = set()
    for r in rows:
        if r["path"] in listed:
            problems.append(f"duplicate MANIFEST row: {r['path']}")
        listed.add(r["path"])
        p = root / r["path"]
        if not p.exists():
            problems.append(f"missing file: {r['path']}")
        elif p.stat().st_size != int(r["bytes"]) or sha256_file(p) != r["sha256"]:
            problems.append(f"hash mismatch: {r['path']}")
    for p in root.rglob("*"):
        if p.is_file() and p.name not in (MANIFEST_NAME, ".gitkeep"):
            rel = p.relative_to(root).as_posix()
            if rel not in listed:
                problems.append(f"not in MANIFEST: {rel}")
    return problems


def latest_dated(root: Path, pattern: str) -> Path | None:
    """Latest file matching a glob whose name ends in _dlYYYYMMDD.<ext> (lexical order = date order)."""
    found = sorted(root.glob(pattern))
    return found[-1] if found else None


def today_utc() -> dt.date:
    return dt.datetime.now(dt.UTC).date()
