"""Divergence-free spherical elementary current systems (SECS), ground horizontal field (D02, D09).

Basis (Amm & Viljanen 1999, Earth Planets Space 51, 431-440; field below the current shell): a
DF SECS of unit amplitude at the ionospheric radius R_I produces at ground radius r, at angular
distance theta from its pole, a horizontal field along the great circle through the pole,

    B_theta = -(mu0 / (4 pi r sin theta)) [(s - cos theta) / sqrt(1 - 2 s cos theta + s^2) + cos theta],

with s = r / R_I and theta-hat pointing away from the pole. B_r is not used (Z is outside P1's
main line, D09). For theta << 1 this tends to the flat-sheet result
mu0 / (4 pi rho) (1 - h / sqrt(rho^2 + h^2)), which the tests check.

Fitting: per minute, the amplitudes are fitted to the valid neighbours' X and Y by truncated SVD,
dropping singular values below epsilon times the largest. The prediction at the target is a
linear operator of the observations, cached per set of valid neighbours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import geo
from .baselines import components

MU0 = 4e-7 * np.pi


def b_theta(theta, r_km: float, ri_km: float) -> np.ndarray:
    """B_theta in nT at ground radius r_km for a 1 A DF SECS at radius ri_km (r_km < ri_km)."""
    theta = np.asarray(theta, float)
    s = r_km / ri_km
    ct, st = np.cos(theta), np.sin(theta)
    root = np.sqrt(1.0 - 2.0 * s * ct + s**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = -(MU0 / (4 * np.pi * r_km * 1e3 * st)) * ((s - ct) / root + ct)
    return np.where(st < 1e-12, 0.0, b) * 1e9


def pole_grid(
    lat: tuple[float, float, float], lon: tuple[float, float, float]
) -> pd.DataFrame:
    """Regular grid of poles; lat and lon are (first, last, step) in degrees, both ends included."""
    lats = np.arange(lat[0], lat[1] + lat[2] / 2, lat[2])
    lons = np.arange(lon[0], lon[1] + lon[2] / 2, lon[2])
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    return pd.DataFrame({"lat": la.ravel(), "lon": lo.ravel()})


def design(
    st_lat, st_lon, pole_lat, pole_lon, height_km: float, r_km: float = geo.R_EARTH_KM
):
    """(Gx, Gy), each (stations x poles): ground X (north) and Y (east) in nT per 1 A amplitude."""
    st_lat, st_lon = (
        np.asarray(st_lat, float)[:, None],
        np.asarray(st_lon, float)[:, None],
    )
    pole_lat, pole_lon = (
        np.asarray(pole_lat, float)[None, :],
        np.asarray(pole_lon, float)[None, :],
    )
    theta = geo.central_angle(st_lat, st_lon, pole_lat, pole_lon)
    az = geo.bearing(st_lat, st_lon, pole_lat, pole_lon)  # towards the pole
    bt = b_theta(
        theta, r_km, r_km + height_km
    )  # along the direction away from the pole
    return -bt * np.cos(az), -bt * np.sin(az)


class SECS:
    name = "secs"

    def __init__(
        self,
        coords: pd.DataFrame,
        poles: pd.DataFrame,
        height_km: float = 110.0,
        epsilon: float = 0.05,
    ):
        self.epsilon = epsilon
        self.height_km = height_km
        self.row = {s: i for i, s in enumerate(coords.index)}
        self.gx, self.gy = design(
            coords.lat, coords.lon, poles.lat, poles.lon, height_km
        )
        self._ops: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}

    def solve(self, valid) -> np.ndarray:
        """Truncated-SVD inverse (poles x 2k) for observations ordered [X_1..X_k, Y_1..Y_k]."""
        idx = [self.row[s] for s in valid]
        g = np.vstack([self.gx[idx], self.gy[idx]])
        u, sv, vt = np.linalg.svd(g, full_matrices=False)
        keep = sv >= self.epsilon * sv[0]
        return vt[keep].T @ np.diag(1.0 / sv[keep]) @ u[:, keep].T

    def operator(self, target: str, valid: tuple[str, ...]) -> np.ndarray:
        """(2 x 2k) map from the valid neighbours' [X..., Y...] to the target's [X, Y]."""
        key = (target, valid)
        if key not in self._ops:
            t = self.row[target]
            gt = np.vstack([self.gx[t], self.gy[t]])
            self._ops[key] = gt @ self.solve(valid)
        return self._ops[key]

    def predict(self, inputs) -> pd.DataFrame:
        sc = inputs.scenario
        neigh = np.array(sc.neighbours)
        x, y = components(inputs.current, neigh)
        ok = ~(np.isnan(x) | np.isnan(y))
        out = np.full((len(x), 2), np.nan)
        patterns, inverse = np.unique(ok, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        for k, pat in enumerate(patterns):
            if not pat.any():
                continue
            rows = inverse == k
            op = self.operator(sc.target, tuple(neigh[pat]))
            obs = np.hstack([x[rows][:, pat], y[rows][:, pat]])
            out[rows] = obs @ op.T
        return pd.DataFrame(out, index=inputs.current.index, columns=["X", "Y"])
