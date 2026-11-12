"""Spectral preprocessing shared by the local spectra and the OSSL libraries."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from utils import C

GRID = np.arange(C.WL_MIN, C.WL_MAX + 1, C.WL_STEP, dtype=float)
PREPROCESSINGS = ("raw", "snv", "sg1d")


def smooth_to_grid(wl: np.ndarray, refl: np.ndarray, window_nm: float = 21.0) -> np.ndarray:
    """Savitzky-Golay smoothing (window ~21 nm, 2nd order) then linear resampling."""
    step = float(np.median(np.diff(wl)))
    win = int(round(window_nm / step))
    win += (win + 1) % 2
    sm = savgol_filter(refl, win, C.SG_POLY, axis=1)
    out = np.empty((refl.shape[0], GRID.size))
    for i in range(refl.shape[0]):
        out[i] = np.interp(GRID, wl, sm[i])
    return out


def snv(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / sd


def sg_derivative(x: np.ndarray) -> np.ndarray:
    return savgol_filter(x, 5, 2, deriv=1, delta=C.WL_STEP, axis=1)


def water_mask() -> np.ndarray:
    keep = np.ones(GRID.size, bool)
    for lo, hi in C.WATER_BANDS:
        keep &= ~((GRID >= lo) & (GRID <= hi))
    return keep


def preprocess(grid_refl: np.ndarray, kind: str, drop_water: bool = False) -> np.ndarray:
    """grid_refl: smoothed reflectance (fraction) on GRID."""
    if kind == "raw":
        x = grid_refl
    elif kind == "snv":
        x = snv(grid_refl)
    elif kind == "sg1d":
        x = sg_derivative(grid_refl)
    else:
        raise ValueError(kind)
    if drop_water:
        x = x[:, water_mask()]
    return x
