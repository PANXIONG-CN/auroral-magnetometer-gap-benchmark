"""Baseline reconstructions (docs/01_research_design.md, "基线方法"; D02).

Every method maps ScenarioInputs to a DataFrame with columns X and Y for the target station over
the scenario minutes, NaN where it gives no output. Spatial methods use only the current minute
of the valid neighbours (protocol section 4); a neighbour is valid in a minute when both X and Y
are present.

- IDW: weights 1/d^p with d the great-circle distance to the target.
- NearestNeighbour: the nearest valid neighbour's values (the limit of IDW as p grows).
- Ridge: linear map from the input stations to the target, fitted on the training period. With
  dynamic=False it gives output only in minutes where all of its input stations are valid; with
  dynamic=True each minute uses a map fitted (same alpha) on exactly the valid input stations, so
  it follows the minute-by-minute neighbour rule of protocol section 3. When the training period
  has too few complete minutes for a combination, the station with the fewest training minutes
  (jointly with the target) is dropped from it until a map can be fitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge as _SkRidge

from . import geo


def components(current: pd.DataFrame, stations) -> tuple[np.ndarray, np.ndarray]:
    """X and Y of the given stations as (minutes x stations) arrays."""
    x = current.xs("X", axis=1, level="component").reindex(columns=list(stations))
    y = current.xs("Y", axis=1, level="component").reindex(columns=list(stations))
    return x.to_numpy(float), y.to_numpy(float)


class IDW:
    name = "idw"

    def __init__(self, coords: pd.DataFrame, power: float = 2.0):
        self.coords = coords
        self.power = power

    def predict(self, inputs) -> pd.DataFrame:
        sc = inputs.scenario
        neigh = list(sc.neighbours)
        t = self.coords.loc[sc.target]
        d = geo.distance_km(
            t.lat, t.lon, self.coords.loc[neigh, "lat"], self.coords.loc[neigh, "lon"]
        )
        w = 1.0 / np.asarray(d, float) ** self.power
        x, y = components(inputs.current, neigh)
        ok = ~(np.isnan(x) | np.isnan(y))
        wsum = ok @ w
        with np.errstate(invalid="ignore", divide="ignore"):
            px = np.where(ok, x, 0.0) @ w / wsum
            py = np.where(ok, y, 0.0) @ w / wsum
        px[wsum == 0] = np.nan
        py[wsum == 0] = np.nan
        return pd.DataFrame({"X": px, "Y": py}, index=inputs.current.index)


class NearestNeighbour:
    name = "nearest"

    def __init__(self, coords: pd.DataFrame):
        self.coords = coords

    def predict(self, inputs) -> pd.DataFrame:
        sc = inputs.scenario
        neigh = list(sc.neighbours)
        t = self.coords.loc[sc.target]
        d = np.asarray(
            geo.distance_km(
                t.lat,
                t.lon,
                self.coords.loc[neigh, "lat"],
                self.coords.loc[neigh, "lon"],
            ),
            float,
        )
        x, y = components(inputs.current, neigh)
        ok = ~(np.isnan(x) | np.isnan(y))
        dist = np.where(ok, d[None, :], np.inf)
        pick = dist.argmin(axis=1)
        rows = np.arange(len(x))
        px, py = x[rows, pick], y[rows, pick]
        none = ~ok.any(axis=1)
        px[none] = np.nan
        py[none] = np.nan
        return pd.DataFrame({"X": px, "Y": py}, index=inputs.current.index)


class RidgeMap:
    """Ridge regression from the input stations' X, Y to the target's X, Y."""

    name = "ridge"

    def __init__(self, alpha: float = 1.0, *, dynamic: bool = False):
        self.alpha = alpha
        self.dynamic = dynamic
        self.models: dict[tuple[str, tuple[str, ...]], _SkRidge] = {}
        self._train: dict[str, pd.DataFrame] = {}
        self._subsets: dict[tuple, tuple[str, ...]] = {}

    @staticmethod
    def _features(data: pd.DataFrame, stations) -> np.ndarray:
        x, y = components(data, stations)
        return np.hstack([x, y])

    def fit(self, train: pd.DataFrame, target: str, inputs) -> RidgeMap:
        """Fit on disturbances `train` (columns (station, component)) in minutes where the target
        and every input station have X and Y. The training data are kept for the per-pattern maps
        of dynamic mode."""
        self._train[target] = train
        if self.dynamic:
            self._usable(target, tuple(inputs))
        else:
            self._fit_map(train, target, tuple(inputs))
        return self

    def _fit_map(
        self, train: pd.DataFrame, target: str, inputs: tuple[str, ...]
    ) -> None:
        feats = self._features(train, inputs)
        tx, ty = components(train, [target])
        out = np.hstack([tx, ty])
        ok = ~np.isnan(feats).any(axis=1) & ~np.isnan(out).any(axis=1)
        if ok.sum() < feats.shape[1] + 2:
            raise ValueError(
                f"too few complete training minutes for {target}: {ok.sum()}"
            )
        self.models[(target, inputs)] = _SkRidge(alpha=self.alpha).fit(
            feats[ok], out[ok]
        )

    @classmethod
    def day_cv(cls, train: pd.DataFrame, target: str, inputs, alphas) -> pd.DataFrame:
        """Pooled horizontal RMSE of leave-one-UT-day-out predictions on `train`, per alpha."""
        inputs = tuple(inputs)
        feats = cls._features(train, inputs)
        tx, ty = components(train, [target])
        out = np.hstack([tx, ty])
        ok = ~np.isnan(feats).any(axis=1) & ~np.isnan(out).any(axis=1)
        f, o = feats[ok], out[ok]
        days = train.index.normalize()[ok]
        rows = []
        for a in alphas:
            se, n = 0.0, 0
            for d in days.unique():
                held = days == d
                p = _SkRidge(alpha=a).fit(f[~held], o[~held]).predict(f[held])
                se += float(((p - o[held]) ** 2).sum())
                n += int(held.sum())
            rows.append({"param": a, "rmse_h": float(np.sqrt(se / n)), "n_scored": n})
        return pd.DataFrame(rows)

    def predict(self, inputs) -> pd.DataFrame:
        sc = inputs.scenario
        neigh = tuple(sc.neighbours)
        x, y = components(inputs.current, neigh)
        ok = ~(np.isnan(x) | np.isnan(y))
        pred = np.full((len(x), 2), np.nan)
        if not self.dynamic:
            key = (sc.target, neigh)
            if key not in self.models:
                raise KeyError(f"no ridge map fitted for {sc.target} from {neigh}")
            rows = ok.all(axis=1)
            if rows.any():
                pred[rows] = self.models[key].predict(np.hstack([x[rows], y[rows]]))
            return pd.DataFrame(pred, index=inputs.current.index, columns=["X", "Y"])
        if sc.target not in self._train:
            raise KeyError(f"no training data for {sc.target}; call fit first")
        patterns, inverse = np.unique(ok, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        for k, pat in enumerate(patterns):
            if not pat.any():
                continue
            used = self._usable(sc.target, tuple(np.array(neigh)[pat]))
            if not used:
                continue
            cols = [neigh.index(st) for st in used]
            rows = inverse == k
            feats = np.hstack([x[rows][:, cols], y[rows][:, cols]])
            pred[rows] = self.models[(sc.target, used)].predict(feats)
        return pd.DataFrame(pred, index=inputs.current.index, columns=["X", "Y"])

    def _usable(self, target: str, valid: tuple[str, ...]) -> tuple[str, ...]:
        """The largest subset of `valid` with a fitted map, dropping the stations with the fewest
        training minutes (jointly with the target) first; () if none can be fitted."""
        key = ("usable", target, valid)
        if key in self._subsets:
            return self._subsets[key]
        train = self._train[target]
        tx, ty = components(train, [target])
        t_ok = ~(np.isnan(tx[:, 0]) | np.isnan(ty[:, 0]))
        x, y = components(train, valid)
        counts = ((~(np.isnan(x) | np.isnan(y))) & t_ok[:, None]).sum(axis=0)
        order = [valid[i] for i in np.argsort(counts, kind="stable")]
        used = valid
        while used:
            if (target, used) in self.models:
                break
            try:
                self._fit_map(train, target, used)
                break
            except ValueError:
                drop = next(st for st in order if st in used)
                used = tuple(st for st in used if st != drop)
        self._subsets[key] = used
        return used
