"""Git and config hashes for result file names (CLAUDE.md: metrics_<git>_cfg<hash>.csv)."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from . import paths

# Code and settings that produce results; experiment notes and result files are left out.
TRACKED = [
    "src",
    "scripts",
    "config",
    ":(glob)experiments/**/*.py",
    ":(glob)experiments/**/*.yaml",
]


def git_short_hash() -> str:
    """Short hash of HEAD; '-dirty' if any TRACKED path differs from HEAD (untracked included)."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=paths.ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    head = git("rev-parse", "--short", "HEAD")
    dirty = git("status", "--porcelain", "--", *TRACKED)
    return head + ("-dirty" if dirty else "")


def files_hash(*files) -> str:
    """First 6 hex digits of SHA-256 over the given files (order-independent)."""
    h = hashlib.sha256()
    for f in sorted(Path(f) for f in files):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:6]


def config_hash(*names: str) -> str:
    """files_hash over the named files in config/."""
    return files_hash(*(paths.CONFIG / n for n in names))


def result_stem(stem: str, *config_files: str) -> str:
    """'<stem>_<git>_cfg<hash>' for a result file; config_files are names in config/."""
    return f"{stem}_{git_short_hash()}_cfg{config_hash(*config_files)}"
