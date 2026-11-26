"""Temporal information added to SECS (docs/01_research_design.md, "时间信息"; D08; claim C3).

All three methods see the scenario's history: neighbours and target in the minutes before the gap
(protocol section 4). Inside the gap they use the neighbours' current minutes only.

- ResidualCorrection(statistic="last"): persistence. SECS plus the target-minus-SECS residual at
  the last valid minute before the gap.
- ResidualCorrection(statistic="mean"): pre-gap bias correction. SECS plus the mean residual over
  the last `window_min` minutes before the gap; the window length is chosen on the validation
  period.
- KalmanSECS: first-order linear state transition of the SECS amplitudes with a Kalman update.
  The state lives in the row space of the design matrix of the neighbours and the target, which
  holds every amplitude combination that any of these stations can see, so the reduction loses
  nothing. In units where the stationary prior variance of the normalised state is 1:

      z_t = a z_(t-1) + w,  w ~ N(0, (1 - a^2) I);    y_t = H_t z_t + v,  v ~ N(0, lam I),

  with H = U S / s_max from the SVD of the design matrix and y in nT. Before the gap the target
  is observed; in the gap only the neighbours are. a = 0 gives a static Tikhonov SECS with
  relative parameter lam, so a measures what memory adds. The filter starts from the prior
  `history_min` minutes before the gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import secs
from .baselines import components
from .scenarios import ScenarioInputs


def _residual_inputs(inputs: ScenarioInputs) -> ScenarioInputs:
    """Inputs whose `current` is the neighbours' history, for the base method over the history."""
    sc = inputs.scenario
    cols = inputs.history.columns
    neigh = cols[cols.get_level_values("station").isin(sc.neighbours)]
    return ScenarioInputs(sc, inputs.history[neigh], inputs.history.iloc[:0])


class ResidualCorrection:
    """Base prediction plus the target-minus-base residual from before the gap."""

    def __init__(self, base, *, window_min: int, statistic: str = "mean"):
        if statistic not in ("mean", "last"):
            raise ValueError("statistic must be 'mean' or 'last'")
        self.base = base
        self.window_min = window_min
        self.statistic = statistic
        self.name = "persistence" if statistic == "last" else "bias_correction"

    def correction(self, inputs: ScenarioInputs) -> np.ndarray:
        """(dX, dY) from the last window_min minutes of history; zeros if none is valid."""
        hist = inputs.history.iloc[-self.window_min :] if self.window_min else None
        if hist is None or hist.empty:
            return np.zeros(2)
        sub = ScenarioInputs(inputs.scenario, inputs.current.iloc[:0], hist)
        base = self.base.predict(_residual_inputs(sub))
        tx, ty = components(hist, [inputs.scenario.target])
        r = np.column_stack([tx[:, 0], ty[:, 0]]) - base[["X", "Y"]].to_numpy(float)
        ok = ~np.isnan(r).any(axis=1)
        if not ok.any():
            return np.zeros(2)
        return r[ok][-1] if self.statistic == "last" else r[ok].mean(axis=0)

    def predict(self, inputs: ScenarioInputs) -> pd.DataFrame:
        return self.base.predict(inputs) + self.correction(inputs)


class KalmanSECS:
    """SECS amplitudes with a first-order linear state transition and a Kalman update."""

    name = "kalman"

    def __init__(
        self,
        coords: pd.DataFrame,
        poles: pd.DataFrame,
        height_km: float = 110.0,
        *,
        a=0.99,
        lam=0.0025,
    ):
        self.a = np.atleast_1d(np.asarray(a, float))
        self.lam = np.atleast_1d(np.asarray(lam, float))
        if self.a.shape != self.lam.shape:
            raise ValueError("a and lam must have the same length")
        if np.any((self.a < 0) | (self.a >= 1)) or np.any(self.lam <= 0):
            raise ValueError("need 0 <= a < 1 and lam > 0")
        self.row = {s: i for i, s in enumerate(coords.index)}
        self.gx, self.gy = secs.design(
            coords.lat, coords.lon, poles.lat, poles.lon, height_km
        )
        self._basis: dict[tuple[str, ...], np.ndarray] = {}

    def basis(self, stations: tuple[str, ...]) -> np.ndarray:
        """H (2m x r): rows X_1..X_m, Y_1..Y_m of the stations in the normalised row space."""
        if stations not in self._basis:
            idx = [self.row[s] for s in stations]
            g = np.vstack([self.gx[idx], self.gy[idx]])
            u, sv, _ = np.linalg.svd(g, full_matrices=False)
            keep = sv > 1e-9 * sv[0]
            self._basis[stations] = u[:, keep] * (sv[keep] / sv[0])
        return self._basis[stations]

    def run(self, inputs: ScenarioInputs) -> np.ndarray:
        """Target (X, Y) in the gap for every (a, lam) pair: array (pairs x minutes x 2)."""
        sc = inputs.scenario
        stations = (*sc.neighbours, sc.target)
        m = len(stations)
        h = self.basis(stations)
        r = h.shape[1]
        hx, hy = components(inputs.history, stations)
        cx, cy = components(inputs.current, sc.neighbours)
        nan = np.full((len(cx), 1), np.nan)
        x = np.vstack([hx, np.hstack([cx, nan])])
        y = np.vstack([hy, np.hstack([cy, nan])])
        n_hist = len(hx)
        b = len(self.a)
        a = self.a[:, None]
        q = (1.0 - self.a**2)[:, None, None]
        lam = self.lam[:, None, None]
        eye = np.eye(r)
        z = np.zeros((b, r))
        p = np.broadcast_to(eye, (b, r, r)).copy()
        target_rows = [m - 1, 2 * m - 1]
        out = np.full((b, len(cx), 2), np.nan)
        for t in range(len(x)):
            z = a * z
            p = (a[:, :, None] ** 2) * p + q * eye
            ok = ~(np.isnan(x[t]) | np.isnan(y[t]))
            if ok.any():
                rows = np.concatenate([np.flatnonzero(ok), m + np.flatnonzero(ok)])
                ht = h[rows]
                obs = np.concatenate([x[t, ok], y[t, ok]])
                pht = p @ ht.T
                s = ht @ pht + lam * np.eye(len(rows))
                k = np.linalg.solve(s, pht.transpose(0, 2, 1)).transpose(0, 2, 1)
                z = z + np.einsum("brk,bk->br", k, obs[None, :] - z @ ht.T)
                p = p - k @ pht.transpose(0, 2, 1)
                p = 0.5 * (p + p.transpose(0, 2, 1))
            if t >= n_hist and ok.any():
                out[:, t - n_hist] = z @ h[target_rows].T
        return out

    def predict(self, inputs: ScenarioInputs) -> pd.DataFrame:
        if len(self.a) != 1:
            raise ValueError("predict needs a single (a, lam); use run for a grid")
        out = self.run(inputs)[0]
        return pd.DataFrame(out, index=inputs.current.index, columns=["X", "Y"])
