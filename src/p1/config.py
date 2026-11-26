"""Read the YAML files in config/."""

from __future__ import annotations

import yaml

from . import paths


def load(name: str) -> dict:
    """Load config/<name>.yaml."""
    with open(paths.CONFIG / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def station_set(name: str) -> list[str]:
    """Station codes of a named set in config/stations.yaml, e.g. 'leave_one_out_12'."""
    return list(load("stations")[name])
