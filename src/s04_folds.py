"""Step 4 - spatial validation design.

Outer level : five spatially coherent groups (k-means on UTM coordinates, no
              soil values used), leave-one-group-out with a training buffer
              (5 km main; 0 and 10 km sensitivity).
Middle level: leave-one-group-out over the remaining outer groups (same buffer).
Inner level : five k-means clusters inside any training set for cross-fitting.
Blocks      : 20 km UTM squares for block bootstrap of error differences.
Conformal   : rotating fit (3 groups) / calibration (1) / test (1) roles.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "16")

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

from utils import C, get_logger, load_soil, save_json, utm_xy

log = get_logger("s04_folds")


def coords_km() -> tuple[pd.Index, np.ndarray]:
    soil = load_soil()
    x, y = utm_xy(soil.lon, soil.lat)
    return soil.index, np.c_[x, y] / 1000.0


def outer_groups(xy: np.ndarray) -> np.ndarray:
    km = KMeans(C.N_OUTER_GROUPS, n_init=100, random_state=C.SEED).fit(xy)
    order = np.argsort(km.cluster_centers_[:, 0])          # west -> east
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[g] for g in km.labels_])


def buffered_train(xy, train_idx, test_idx, buffer_km):
    if buffer_km <= 0 or len(test_idx) == 0:
        return np.asarray(train_idx)
    d = cdist(xy[train_idx], xy[test_idx]).min(axis=1)
    return np.asarray(train_idx)[d > buffer_km]


def outer_splits(groups, xy, buffer_km):
    out = []
    for g in range(C.N_OUTER_GROUPS):
        te = np.where(groups == g)[0]
        tr = buffered_train(xy, np.where(groups != g)[0], te, buffer_km)
        out.append((tr, te))
    return out


def middle_splits(train_idx, groups, xy, buffer_km):
    """Leave-one-group-out inside an outer training set."""
    out = []
    for g in np.unique(groups[train_idx]):
        va = train_idx[groups[train_idx] == g]
        tr = buffered_train(xy, train_idx[groups[train_idx] != g], va, buffer_km)
        if len(tr) >= 10 and len(va) >= 3:
            out.append((tr, va))
    return out


def middle_splits_kmeans(train_idx, xy, buffer_km, k=5, repeats=1, seed=C.SEED):
    """Spatial grouped CV inside an outer training set (k-means clusters).

    Returns (repeat, train, validation) triples; the validation cluster is buffered.
    Extra repeats with other seeds reproduce the same partition for these data.
    """
    out = []
    kk = int(min(k, max(2, len(train_idx) // 6)))
    for r in range(repeats):
        lab = KMeans(kk, n_init=10, random_state=seed + r).fit_predict(xy[train_idx])
        for g in range(kk):
            va = train_idx[lab == g]
            tr = buffered_train(xy, train_idx[lab != g], va, buffer_km)
            if len(tr) >= 10 and len(va) >= 3:
                out.append((r, tr, va))
    return out


def inner_groups(train_idx, xy, k=5, seed=C.SEED):
    k = min(k, len(train_idx) // 4)
    return KMeans(k, n_init=10, random_state=seed).fit_predict(xy[train_idx])


def blocks(xy, size_km=C.BLOCK_KM):
    bx = np.floor(xy[:, 0] / size_km).astype(int)
    by = np.floor(xy[:, 1] / size_km).astype(int)
    _, bid = np.unique(np.c_[bx, by], axis=0, return_inverse=True)
    return bid.ravel()


def conformal_roles(groups, xy, buffer_km=C.BUFFER_MAIN_KM, all_calibrations=False):
    roles = []
    for t in range(C.N_OUTER_GROUPS):
        cals = range(C.N_OUTER_GROUPS) if all_calibrations else [(t + 1) % C.N_OUTER_GROUPS]
        for c in cals:
            if c == t:
                continue
            test = np.where(groups == t)[0]
            cal = np.where(groups == c)[0]
            fit = np.where((groups != t) & (groups != c))[0]
            fit = buffered_train(xy, fit, np.r_[test, cal], buffer_km)
            roles.append({"test_group": t, "cal_group": c, "fit": fit, "cal": cal, "test": test})
    return roles


def main() -> None:
    ids, xy = coords_km()
    groups = outer_groups(xy)
    rnd = np.empty(len(ids), int)
    for k, (_, te) in enumerate(KFold(5, shuffle=True, random_state=C.SEED).split(xy)):
        rnd[te] = k
    bid = blocks(xy)
    tab = pd.DataFrame({"sample_id": ids, "x_km": xy[:, 0], "y_km": xy[:, 1],
                        "outer_group": groups, "random_fold": rnd, "block20": bid})
    tab.to_csv(C.FOLDS / "folds.csv", index=False)

    design = {"group_sizes": np.bincount(groups).tolist(), "n_blocks": int(bid.max() + 1),
              "buffers": {}}
    for b in C.BUFFER_SENS_KM:
        rows = []
        for g, (tr, te) in enumerate(outer_splits(groups, xy, b)):
            dmin = cdist(xy[te], xy[tr]).min(axis=1)
            removed = int((groups != g).sum() - len(tr))
            rows.append({"group": g, "n_test": len(te), "n_train": len(tr),
                         "n_removed_by_buffer": removed,
                         "min_test_train_km": float(dmin.min()),
                         "median_test_train_km": float(np.median(dmin))})
        design["buffers"][str(b)] = rows
        pd.DataFrame(rows).to_csv(C.FOLDS / f"outer_design_buffer{int(b)}km.csv", index=False)
    cr = conformal_roles(groups, xy)
    design["conformal"] = [{"test_group": r["test_group"], "cal_group": r["cal_group"],
                            "n_fit": len(r["fit"]), "n_cal": len(r["cal"]),
                            "n_test": len(r["test"])} for r in cr]
    save_json(design, C.FOLDS / "design_summary.json")
    log.info("groups (W->E) sizes %s; blocks %d", design["group_sizes"], design["n_blocks"])
    for b, rows in design["buffers"].items():
        log.info("buffer %s km: %s", b, [(r["n_test"], r["n_train"]) for r in rows])


if __name__ == "__main__":
    main()
