"""Scenario metrics (docs/03_evaluation_protocol.md section 5).

Inputs are disturbances (observed minus reference): DataFrames indexed by minute with columns
'X' and 'Y' in nT. A minute is scored when X and Y are present in both the prediction and the
truth, and in `mask` if one is given. Methods compared in one table must be scored on the same
minutes (CLAUDE.md hard rule 4): pass them all the mask from `common_mask`.

- rmse_h: sqrt(mean((X^ - X)^2 + (Y^ - Y)^2)) over scored minutes; rmse_x, rmse_y per component.
- peak_err: |max B_H^ - max B_H| over scored minutes, with B_H = sqrt(X^2 + Y^2).
- peak_dt_min: time of max B_H^ minus time of max B_H, in minutes (first maximum on ties).
- dbdt_rmse: RMSE of dB_H/dt = sqrt((X_t - X_t-1)^2 + (Y_t - Y_t-1)^2), in nT/min, over minutes
  t where t and t-1 are both scored and one minute apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MINUTE = pd.Timedelta(minutes=1)
NAN = float("nan")


def _xy(df: pd.DataFrame, index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    d = df.reindex(index)
    return d["X"].to_numpy(float), d["Y"].to_numpy(float)


def _ok(df: pd.DataFrame, index: pd.Index) -> np.ndarray:
    x, y = _xy(df, index)
    return ~(np.isnan(x) | np.isnan(y))


def common_mask(truth: pd.DataFrame, *preds: pd.DataFrame) -> pd.Series:
    """Minutes where the truth and every prediction have X and Y."""
    ok = _ok(truth, truth.index)
    for p in preds:
        ok &= _ok(p, truth.index)
    return pd.Series(ok, index=truth.index)


def score(
    pred: pd.DataFrame, truth: pd.DataFrame, mask: pd.Series | None = None
) -> dict:
    """Metrics of one prediction against the hidden truth of one scenario."""
    idx = truth.index
    px, py = _xy(pred, idx)
    tx, ty = _xy(truth, idx)
    ok = ~(np.isnan(px) | np.isnan(py) | np.isnan(tx) | np.isnan(ty))
    if mask is not None:
        ok &= mask.reindex(idx, fill_value=False).to_numpy(bool)
    out = {
        "n_scored": int(ok.sum()),
        "rmse_h": NAN,
        "rmse_x": NAN,
        "rmse_y": NAN,
        "peak_err": NAN,
        "peak_dt_min": NAN,
        "n_dbdt": 0,
        "dbdt_rmse": NAN,
    }
    if not ok.any():
        return out

    dx, dy = (px - tx)[ok], (py - ty)[ok]
    out["rmse_x"] = float(np.sqrt(np.mean(dx**2)))
    out["rmse_y"] = float(np.sqrt(np.mean(dy**2)))
    out["rmse_h"] = float(np.sqrt(np.mean(dx**2 + dy**2)))

    bh_p, bh_t = np.hypot(px, py)[ok], np.hypot(tx, ty)[ok]
    times = idx[ok]
    ip, it = int(np.argmax(bh_p)), int(np.argmax(bh_t))
    out["peak_err"] = float(abs(bh_p[ip] - bh_t[it]))
    out["peak_dt_min"] = float((times[ip] - times[it]) / MINUTE)

    step = ok[1:] & ok[:-1] & (np.diff(idx.values) == np.timedelta64(1, "m"))
    if step.any():
        d_p = np.hypot(np.diff(px), np.diff(py))[step]
        d_t = np.hypot(np.diff(tx), np.diff(ty))[step]
        out["n_dbdt"] = int(step.sum())
        out["dbdt_rmse"] = float(np.sqrt(np.mean((d_p - d_t) ** 2)))
    return out
