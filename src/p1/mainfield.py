"""Main field from CHAOS or IGRF at station positions, in geodetic X (north), Y (east), Z (down), nT.

core: the time-dependent internal field of the model (secular variation included); static: the
lithospheric field, constant in time. Stations sit on the WGS84 ellipsoid at height 0 (the IMAGE
table gives no heights). Times outside a model's spline range are extrapolated linearly
(chaosmagpy default), which is what a user of that release would have had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("2000-01-01")


def mjd2000(times) -> np.ndarray:
    """Days since 2000-01-01 00:00 UT."""
    return np.asarray(
        (pd.DatetimeIndex(np.atleast_1d(times)) - EPOCH) / pd.Timedelta(days=1), float
    )


def _geocentric(coords: pd.DataFrame):
    from chaosmagpy import coordinate_utils as cu

    radius, theta = cu.gg_to_geo(
        np.zeros(len(coords)), 90.0 - coords["lat"].to_numpy(float)
    )
    return radius, theta, coords["lon"].to_numpy(float)


def _to_geodetic(radius, theta, b_r, b_t, b_p):
    from chaosmagpy import coordinate_utils as cu

    _, _, x, z = cu.geo_to_gg(radius, theta, b_r, b_t)
    return x, b_p, z


def core_xyz(model, coords: pd.DataFrame, times) -> pd.DataFrame:
    """Core field at each station and time: index = times, columns = (station, component)."""
    t = mjd2000(times)
    radius, theta, phi = _geocentric(coords)
    shape = (len(t), len(coords))
    rr, tt, pp = (np.broadcast_to(a, shape) for a in (radius, theta, phi))
    tm = np.broadcast_to(t[:, None], shape)
    # CHAOS splits core and crust (synth_values_tdep); an IGRF model is core only
    synth = getattr(model, "synth_values_tdep", None) or model.synth_values
    b_r, b_t, b_p = synth(tm, rr, tt, pp)
    x, y, z = _to_geodetic(rr, tt, b_r, b_t, b_p)
    cols = pd.MultiIndex.from_product(
        [coords.index, ["X", "Y", "Z"]], names=["station", "component"]
    )
    data = np.stack([x, y, z], axis=2).reshape(len(t), -1)
    return pd.DataFrame(
        data, index=pd.DatetimeIndex(np.atleast_1d(times), name="time"), columns=cols
    )


def static_xyz(model, coords: pd.DataFrame) -> pd.DataFrame:
    """Lithospheric (static) field of the model at each station: index = station, columns X, Y, Z."""
    radius, theta, phi = _geocentric(coords)
    b_r, b_t, b_p = model.synth_values_static(radius, theta, phi)
    x, y, z = _to_geodetic(radius, theta, b_r, b_t, b_p)
    return pd.DataFrame({"X": x, "Y": y, "Z": z}, index=coords.index)
