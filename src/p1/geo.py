"""Spherical geometry on the ground sphere and station coordinates.

Coordinates are geographic latitude and longitude in degrees; station coordinates come from the
IMAGE station table at run time (config/stations.yaml says not to type them in).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

R_EARTH_KM = 6371.2  # geomagnetic reference radius


def station_coords(stations) -> pd.DataFrame:
    """lat, lon of the given stations from the cached IMAGE station table."""
    from .io import image

    table = image.read_station_table()
    missing = sorted(set(stations) - set(table.index))
    if missing:
        raise KeyError(f"stations not in the IMAGE table: {missing}")
    return table.loc[list(stations), ["lat", "lon"]].astype(float)


def central_angle(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle angle in radians (haversine; arguments in degrees, broadcastable)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def distance_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    return R_EARTH_KM * central_angle(lat1, lon1, lat2, lon2)


def bearing(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Azimuth at point 1 towards point 2, radians clockwise from geographic north."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    return np.arctan2(
        np.sin(dl) * np.cos(p2),
        np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl),
    )
