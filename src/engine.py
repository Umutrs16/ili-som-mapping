"""Nested spatial cross-validation engine, metrics and block bootstrap.

Outer : leave-one-spatial-group-out with a training buffer (test labels and
        spectra never enter any fit).
Middle: spatial grouped CV inside the outer training set (five k-means
        clusters, same buffer) selects hyper-parameters, latent dimension and
        target scale by pooled RMSE.
Inner : spatial cross-fitting inside transfer_fit_predict (models.py).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

import models as M
from s04_folds import middle_splits_kmeans, outer_splits
from utils import C


@lru_cache(maxsize=24)
def get_data(variant="main_s20_buffer", opt_variant=None, ae_variant=None, target=C.TARGET):
    """Per-process cache so parallel tasks only exchange indices, not arrays."""
    return M.load_data(variant, opt_variant, ae_variant, target)


@lru_cache(maxsize=1)
def get_encoders():
    from encoders import load_public_encoders
    return load_public_encoders()


def data_key(variant="main_s20_buffer", opt_variant=None, ae_variant=None, target=C.TARGET):
    return (variant, opt_variant, ae_variant, target)


def load_run(tag: str):
    import pickle
    p = C.OOF / f"{tag}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


@dataclass
class Experiment:
    name: str
    kind: str                      # mean | trend | kriging | student | transfer | lab_full | lab_sim
    groups: list = field(default_factory=list)
    student: str = "ridge"
    encoder: str | None = None     # 'local' | 'ICRAF-ISRIC' | 'LUCAS' | 'Pooled'
    prep: str = "snv"
    drop_water: bool = False
    dims: tuple = tuple(C.LATENT_DIMS)
    shuffle: bool = False
    true_latent: bool = False
    log_target: bool | None = None  # None: log vs. original scale chosen in the middle CV
    ae_pca: int | None = None
    label: str = ""

    def grid(self):
        if self.kind == "mean" or self.kind == "kriging":
            return [{}]
        if self.kind == "trend":
            base = M.GRIDS["ridge"]
        elif self.kind == "lab_full":
            # Fair comparison with lab_sim: same prep candidates; PLSR tunes n, ridge uses its α grid.
            if self.student == "pls":
                base = [{"n": n, "prep": p} for n in (2, 5, 10) for p in ("raw", "snv", "sg1d")]
            else:
                base = [dict(b, prep=p) for p in ("raw", "snv", "sg1d") for b in M.GRIDS[self.student]]
        else:
            base = M.GRIDS[self.student]
            if self.kind == "transfer":
                base = [dict(b, d=d) for d in self.dims for b in base]
        scales = (True, False) if self.log_target is None else (self.log_target,)
        return [dict(b, log=s) for s in scales for b in base]


def _X(exp: Experiment, D: M.Data):
    if exp.kind == "lab_sim":
        return D.sim.to_numpy(float)
    return D.X[M.features_for(exp.groups)].to_numpy(float)


class AEPCA:
    """In-fold PCA compression of the 64 AlphaEarth dimensions (sensitivity)."""

    def __init__(self, cols, k):
        self.cols, self.k = cols, k

    def fit_transform(self, X_tr, X_te):
        from sklearn.decomposition import PCA
        a = np.isin(np.arange(X_tr.shape[1]), self.cols)
        p = PCA(self.k, random_state=C.SEED).fit(X_tr[:, a])
        return (np.c_[X_tr[:, ~a], p.transform(X_tr[:, a])],
                np.c_[X_te[:, ~a], p.transform(X_te[:, a])])


def fit_predict(exp: Experiment, theta: dict, tr, te, dk: tuple,
                seed: int = C.SEED, return_latent: bool = False):
    D = get_data(*dk)
    encoders = get_encoders() if exp.kind == "transfer" and exp.encoder != "local" else {}
    y = D.y
    log = theta.get("log", True)
    theta = {k: v for k, v in theta.items() if k != "log"}
    if exp.kind == "mean":
        return np.full(len(te), y[tr].mean())
    if exp.kind == "trend":
        return M.trend_fit_predict(D.xy[tr], y[tr], D.xy[te], theta["alpha"], log)
    if exp.kind == "kriging":
        try:
            return M.kriging_fit_predict(D.xy[tr], y[tr], D.xy[te])
        except Exception:
            return np.full(len(te), y[tr].mean())
    if exp.kind == "lab_full":
        S = D.spectra[(theta["prep"], False)]
        p = {k: v for k, v in theta.items() if k not in ("prep", "log")}
        est = M.LogTarget(M.make_student(exp.student, p, seed), log)
        return est.fit(S[tr], y[tr]).predict(S[te])
    X = _X(exp, D)
    X_tr, X_te = X[tr], X[te]
    if exp.ae_pca:
        cols = [i for i, c in enumerate(M.features_for(exp.groups)) if c.startswith("AE")]
        X_tr, X_te = AEPCA(cols, exp.ae_pca).fit_transform(X_tr, X_te)
    if exp.kind in ("student", "lab_sim"):
        est = M.LogTarget(M.make_student(exp.student, theta, seed), log)
        return est.fit(X_tr, y[tr]).predict(X_te)
    if exp.kind == "transfer":
        S = D.spectra[(exp.prep, exp.drop_water)]
        d = theta["d"]
        p = {k: v for k, v in theta.items() if k != "d"}
        if exp.encoder == "local":
            enc = M.LocalPCA(d)
        else:
            enc = M.FixedPCA(encoders[(exp.encoder, exp.prep, exp.drop_water)], d)
        return M.transfer_fit_predict(exp.student, p, enc, X_tr, y[tr], S[tr], D.xy[tr], X_te,
                                      S_te=S[te], shuffle=exp.shuffle,
                                      true_latent=exp.true_latent, log_target=log,
                                      seed=seed, return_latent=return_latent)
    raise ValueError(exp.kind)


class Fitted:
    """A fitted deployable model working on the experiment's covariate matrix."""

    def __init__(self, exp, model, features, theta):
        self.exp, self.model, self.features, self.theta = exp, model, features, theta

    def predict(self, X):
        return self.model.predict(np.asarray(X, float))


def fit_model(exp: Experiment, theta: dict, tr, dk: tuple = data_key(),
              seed: int = C.SEED) -> Fitted:
    """Same computations as fit_predict, but returns the fitted model (student/transfer)."""
    D = get_data(*dk)
    log = theta.get("log", True)
    th = {k: v for k, v in theta.items() if k != "log"}
    feats = M.features_for(exp.groups)
    X = D.X[feats].to_numpy(float)
    if exp.kind == "student" and not exp.ae_pca:
        m = M.LogTarget(M.make_student(exp.student, th, seed), log).fit(X[tr], D.y[tr])
    elif exp.kind == "transfer" and not exp.true_latent:
        S = D.spectra[(exp.prep, exp.drop_water)]
        p = {k: v for k, v in th.items() if k != "d"}
        if exp.encoder == "local":
            enc = M.LocalPCA(th["d"])
        else:
            enc = M.FixedPCA(get_encoders()[(exp.encoder, exp.prep, exp.drop_water)], th["d"])
        m = M.TransferModel(exp.student, p, enc, log, exp.shuffle, seed).fit(
            X[tr], D.y[tr], S[tr], D.xy[tr])
    else:
        raise ValueError(f"not deployable: {exp.kind}")
    return Fitted(exp, m, feats, theta)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    r = np.corrcoef(y, p)[0, 1] if np.std(p) > 0 else 0.0
    ccc = 2 * np.cov(y, p, bias=True)[0, 1] / (y.var() + p.var() + (y.mean() - p.mean()) ** 2)
    return {"r2": 1 - ((y - p) ** 2).sum() / ss, "rmse": rmse(y, p),
            "mae": float(np.mean(np.abs(y - p))), "bias": float(np.mean(p - y)),
            "r": float(r), "ccc": float(ccc), "n": int(len(y))}


def _middle_task(exp, theta, tr_mid, va_mid, dk, seed=C.SEED):
    return va_mid, fit_predict(exp, theta, tr_mid, va_mid, dk, seed=seed)


def _par(calls, nj):
    if nj == 1:
        return [f(*a, **k) for f, a, k in calls]
    return Parallel(n_jobs=nj, backend="loky")(delayed(f)(*a, **k) for f, a, k in calls)


def select_params(exp: Experiment, train_sets: list, dk: tuple = data_key(),
                  buffer_km: float = C.BUFFER_MAIN_KM, scheme: str = "spatial",
                  n_jobs: int = 14, seed: int = C.SEED) -> list[dict]:
    """Middle-level grouped CV inside each training set; returns one theta per set."""
    D = get_data(*dk)
    grid = exp.grid()
    if len(grid) == 1:
        return [grid[0]] * len(train_sets)
    mids = []
    for tr in train_sets:
        if scheme == "spatial":
            mids.append(middle_splits_kmeans(tr, D.xy, buffer_km))
        else:
            mids.append([(0, tr[a], tr[b])
                         for a, b in KFold(5, shuffle=True, random_state=C.SEED).split(tr)])
    tasks = [(i, gi, j) for i in range(len(train_sets)) for gi in range(len(grid))
             for j in range(len(mids[i]))]
    res = _par([(_middle_task, (exp, grid[gi], mids[i][j][1], mids[i][j][2], dk),
                 {"seed": seed}) for i, gi, j in tasks], n_jobs)
    sq = {}
    for (i, gi, j), (va, pv) in zip(tasks, res):
        sq.setdefault((i, gi, mids[i][j][0]), []).append((D.y[va] - pv) ** 2)
    chosen = []
    for i in range(len(train_sets)):
        vals = []
        for gi in range(len(grid)):
            reps = [np.sqrt(np.mean(np.concatenate(v))) for (a, b, _), v in sq.items()
                    if a == i and b == gi]
            vals.append(np.mean(reps) if reps else np.inf)
        chosen.append(grid[int(np.argmin(vals))] if np.isfinite(min(vals))
                      else grid[len(grid) // 2])
    return chosen


def run(exp: Experiment, dk: tuple = data_key(), buffer_km: float = C.BUFFER_MAIN_KM,
        scheme: str = "spatial", n_jobs: int = 14, train_subsets: dict | None = None,
        record_latent: bool = False, seed: int = C.SEED) -> dict:
    """Nested CV. train_subsets: optional {outer_fold: train_idx} override (learning curves).

    n_jobs=1 runs sequentially (used when many runs are parallelised externally).
    """
    t0 = time.time()
    D = get_data(*dk)
    n = len(D.y)
    if scheme == "spatial":
        outer = outer_splits(D.groups, D.xy, buffer_km)
    else:
        outer = [(np.where(D.rnd != k)[0], np.where(D.rnd == k)[0]) for k in range(5)]
    if train_subsets:
        outer = [(train_subsets.get(i, tr), te) for i, (tr, te) in enumerate(outer)]
    jobs = 1 if exp.student == "tabpfn" else n_jobs
    chosen = select_params(exp, [tr for tr, _ in outer], dk, buffer_km, scheme, jobs, seed)
    finals = _par([(fit_predict, (exp, chosen[i], outer[i][0], outer[i][1], dk),
                    {"seed": seed, "return_latent": record_latent and exp.kind == "transfer"})
                   for i in range(len(outer))], min(jobs, len(outer)))
    pred = np.full(n, np.nan)
    latent = {}
    for i, ((tr, te), out) in enumerate(zip(outer, finals)):
        if record_latent and exp.kind == "transfer":
            p, zhat, ztrue = out
            latent[i] = {"te": te, "zhat": zhat, "ztrue": ztrue}
            pred[te] = p
        else:
            pred[te] = out
    mask = ~np.isnan(pred)
    out = {"name": exp.name, "pred": pred, "params": chosen, "scheme": scheme,
           "buffer_km": buffer_km, "metrics": metrics(D.y[mask], pred[mask]),
           "group_metrics": {int(g): metrics(D.y[D.groups == g], pred[D.groups == g])
                             for g in np.unique(D.groups) if np.all(mask[D.groups == g])},
           "n_train": [int(len(tr)) for tr, _ in outer], "seconds": time.time() - t0}
    if latent:
        out["latent"] = latent
    return out


def block_bootstrap_delta(y, pa, pb, blocks, n_boot=C.N_BOOT, seed=C.SEED) -> dict:
    """Paired block bootstrap of RMSE(b) - RMSE(a) and relative change (%)."""
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    idx_by = {b: np.where(blocks == b)[0] for b in ub}
    d_abs, d_rel = np.empty(n_boot), np.empty(n_boot)
    for k in range(n_boot):
        pick = np.concatenate([idx_by[b] for b in rng.choice(ub, size=len(ub), replace=True)])
        ra, rb = rmse(y[pick], pa[pick]), rmse(y[pick], pb[pick])
        d_abs[k], d_rel[k] = rb - ra, 100 * (rb - ra) / ra
    ra, rb = rmse(y, pa), rmse(y, pb)
    return {"delta_rmse": rb - ra, "delta_pct": 100 * (rb - ra) / ra,
            "ci_low": float(np.percentile(d_abs, 2.5)), "ci_high": float(np.percentile(d_abs, 97.5)),
            "pct_ci_low": float(np.percentile(d_rel, 2.5)),
            "pct_ci_high": float(np.percentile(d_rel, 97.5)),
            "p_improve": float(np.mean(d_abs < 0))}


def group_direction(y, pa, pb, groups) -> int:
    """Number of spatial groups in which model b has lower RMSE than model a."""
    return int(sum(rmse(y[groups == g], pb[groups == g]) < rmse(y[groups == g], pa[groups == g])
                   for g in np.unique(groups)))
