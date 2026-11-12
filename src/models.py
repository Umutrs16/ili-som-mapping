"""Data container, feature groups, learners and the cross-fitted spectral transfer.

Every learner is wrapped as  impute(median) -> standardise -> model, fitted on
log(OM) and back-transformed with Duan's smearing factor estimated from the
training residuals only. All transformations are fitted inside the current
training subset.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from s01_gee_features import (AE_NAMES, CLIM_FEATURES, OPT_FEATURES, SAR_FEATURES,
                              SG_FEATURES, TER_FEATURES, VEG_FEATURES)
from spectra import preprocess
from utils import C, load_soil

GROUPS = {"O": OPT_FEATURES, "T": TER_FEATURES, "R": SAR_FEATURES, "V": VEG_FEATURES,
          "C": CLIM_FEATURES, "A": AE_NAMES, "XY": ["x_km", "y_km"], "SG": SG_FEATURES}
GROUP_LABEL = {"O": "Bare-soil optical", "T": "Terrain", "R": "SAR", "V": "Antecedent NDVI",
               "C": "Climate", "A": "AlphaEarth", "XY": "Coordinates", "SG": "SoilGrids",
               "Z": "Spectral latent"}
BASE_SETS = {"E1": ["O", "T"], "E2": ["O", "T", "R", "V", "C"],
             "E3": ["O", "T", "R", "V", "C", "A"]}

GRIDS = {
    "ridge": [{"alpha": a} for a in (0.1, 1, 10, 100, 1000)],
    "pls": [{"n": n} for n in (2, 5, 10)],
    "krr": [{"alpha": a} for a in (0.1, 1, 10)],
    "cat": [{"depth": d, "iterations": it, "l2": l2}
            for d in (3, 4) for it in (200, 500) for l2 in (3, 10)],
    "rf": [{"leaf": lf} for lf in (3, 5, 8)],
    "tabpfn": [{}],
}
STUDENT_LABEL = {"ridge": "Ridge", "pls": "PLSR", "krr": "Kernel ridge", "cat": "CatBoost",
                 "rf": "Random forest", "tabpfn": "TabPFN v2"}


# ------------------------------------------------------------------ data
@dataclass
class Data:
    ids: np.ndarray
    y: np.ndarray
    X: pd.DataFrame
    spectra: dict = field(default_factory=dict)      # (prep, drop_water) -> array
    xy: np.ndarray | None = None
    groups: np.ndarray | None = None
    blocks: np.ndarray | None = None
    rnd: np.ndarray | None = None
    sim: pd.DataFrame | None = None                  # SRF-simulated lab S2 bands
    target: str = C.TARGET


def load_data(variant: str = "main_s20_buffer", opt_variant: str | None = None,
              ae_variant: str | None = None, target: str = C.TARGET) -> Data:
    soil = load_soil()
    X = pd.read_csv(C.GEE_DIR / f"points_{variant}.csv", index_col="sample_id").loc[soil.index]
    main = pd.read_csv(C.GEE_DIR / "points_main_s20_buffer.csv",
                       index_col="sample_id").loc[soil.index]
    for col in set(GROUPS["SG"] + GROUPS["C"]) - set(X.columns):
        X[col] = main[col]
    if opt_variant:
        alt = pd.read_csv(C.GEE_DIR / f"points_{opt_variant}.csv", index_col="sample_id")
        for col in OPT_FEATURES:
            X[col] = alt.loc[soil.index, col]
    if ae_variant:
        alt = pd.read_csv(C.GEE_DIR / f"points_{ae_variant}.csv", index_col="sample_id")
        for col in AE_NAMES:
            X[col] = alt.loc[soil.index, col]
    folds = pd.read_csv(C.FOLDS / "folds.csv", index_col="sample_id").loc[soil.index]
    X["x_km"], X["y_km"] = folds["x_km"], folds["y_km"]
    spec = np.load(C.FEAT / "local_spectra_grid5nm.npz")
    assert (spec["sample_id"] == soil.index.to_numpy()).all()
    spectra = {(p, dw): preprocess(spec["refl"], p, dw)
               for p in ("raw", "snv", "sg1d") for dw in (False, True)}
    sim_path = C.FEAT / "lab_sim_s2_weighted_v4.0.csv"
    sim = pd.read_csv(sim_path, index_col="sample_id").loc[soil.index] if sim_path.exists() else None
    return Data(ids=soil.index.to_numpy(), y=soil[target].to_numpy(float), X=X,
                spectra=spectra, xy=folds[["x_km", "y_km"]].to_numpy(),
                groups=folds["outer_group"].to_numpy(), blocks=folds["block20"].to_numpy(),
                rnd=folds["random_fold"].to_numpy(), sim=sim, target=target)


def features_for(groups: list[str]) -> list[str]:
    cols = []
    for g in groups:
        cols += GROUPS[g]
    return cols


# ------------------------------------------------------------------ learners
class PLSR(BaseEstimator, RegressorMixin):
    def __init__(self, n=5):
        self.n = n

    def fit(self, X, y):
        k = int(max(1, min(self.n, X.shape[0] - 1, X.shape[1])))
        self.m_ = PLSRegression(n_components=k, scale=False).fit(X, y)
        return self

    def predict(self, X):
        return self.m_.predict(X).ravel()


class MedianKRR(BaseEstimator, RegressorMixin):
    """RBF kernel ridge; bandwidth from the median squared distance of the training data."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[np.triu_indices_from(d2, 1)])
        self.gamma_ = 1.0 / max(med, 1e-12)
        self.ym_ = y.mean()
        self.m_ = KernelRidge(alpha=self.alpha, kernel="rbf", gamma=self.gamma_).fit(X, y - self.ym_)
        return self

    def predict(self, X):
        return self.m_.predict(X) + self.ym_


class TabPFNWrap(BaseEstimator, RegressorMixin):
    def __init__(self, seed=C.SEED):
        self.seed = seed

    def fit(self, X, y):
        from tabpfn import TabPFNRegressor
        self.m_ = TabPFNRegressor(model_path=str(C.TABPFN_CKPT), device="cuda",
                                  random_state=self.seed, ignore_pretraining_limits=True)
        self.m_.fit(X, y)
        return self

    def predict(self, X):
        return self.m_.predict(X)


def make_student(name: str, p: dict, seed: int = C.SEED):
    if name == "ridge":
        est = Ridge(alpha=p["alpha"])
    elif name == "pls":
        est = PLSR(n=p["n"])
    elif name == "krr":
        est = MedianKRR(alpha=p["alpha"])
    elif name == "cat":
        from catboost import CatBoostRegressor
        est = CatBoostRegressor(depth=p["depth"], iterations=p["iterations"],
                                l2_leaf_reg=p["l2"], learning_rate=0.05, loss_function="RMSE",
                                random_seed=seed, verbose=0, thread_count=1,
                                allow_writing_files=False)
    elif name == "rf":
        est = RandomForestRegressor(n_estimators=500, min_samples_leaf=p["leaf"],
                                    max_features=0.33, random_state=seed, n_jobs=1)
    elif name == "tabpfn":
        est = TabPFNWrap(seed=seed)
    else:
        raise ValueError(name)
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("m", est)])


class LogTarget(BaseEstimator, RegressorMixin):
    """Fit on log(y); back-transform with Duan's smearing factor (training residuals)."""

    def __init__(self, est, log=True):
        self.est = est
        self.log = log

    def fit(self, X, y):
        self.e_ = clone(self.est)
        if not self.log:
            self.e_.fit(X, y)
            self.smear_ = 1.0
            return self
        ly = np.log(y)
        self.e_.fit(X, ly)
        r = ly - np.asarray(self.e_.predict(X)).ravel()
        self.smear_ = float(np.mean(np.exp(r)))
        return self

    def predict(self, X):
        p = np.asarray(self.e_.predict(X)).ravel()
        return np.exp(p) * self.smear_ if self.log else p


def make_G():
    """Satellite covariates -> spectral latent (multi-output ridge, internal GCV)."""
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("m", RidgeCV(alphas=np.logspace(-2, 3, 11), alpha_per_target=True))])


# ------------------------------------------------------------------ spectral encoders
class LocalPCA:
    """Encoder refitted on the spectra of the current training subset only."""

    def __init__(self, d):
        self.d = d

    def fit(self, S):
        self.p_ = PCA(min(self.d, S.shape[0] - 1), random_state=C.SEED).fit(S)
        return self

    def transform(self, S):
        return self.p_.transform(S)[:, :self.d]


class FixedPCA:
    """Public-library encoder (already fitted on OSSL; never sees local labels)."""

    def __init__(self, pca, d):
        self.pca, self.d = pca, d

    def fit(self, S):
        return self

    def transform(self, S):
        return self.pca.transform(S)[:, :self.d]


def inner_labels(xy_tr: np.ndarray, k: int = 5, seed: int = C.SEED) -> np.ndarray:
    from sklearn.cluster import KMeans
    k = int(min(k, max(2, len(xy_tr) // 6)))
    return KMeans(k, n_init=10, random_state=seed).fit_predict(xy_tr)


class TransferModel:
    """Cross-fitted transfer: z = E(s); G: x -> z; H: [x, z_hat] -> y.

    H is trained on out-of-cluster latent predictions (spatial k-means inside the
    training data), so it sees latent estimates of the same quality as at deployment.
    """

    def __init__(self, student, p, encoder, log_target=True, shuffle=False, seed=C.SEED):
        self.student, self.p, self.encoder = student, p, encoder
        self.log_target, self.shuffle, self.seed = log_target, shuffle, seed

    def fit(self, X, y, S, xy):
        rng = np.random.default_rng(self.seed)
        S_use = S[rng.permutation(len(S))] if self.shuffle else S
        self.enc_ = self.encoder.fit(S_use)
        Z = self.enc_.transform(S_use)
        lab = inner_labels(xy, seed=self.seed)
        Zhat = np.empty_like(Z)
        for k in np.unique(lab):
            a, b = lab != k, lab == k
            Zhat[b] = make_G().fit(X[a], Z[a]).predict(X[b])
        self.H_ = LogTarget(make_student(self.student, self.p, self.seed),
                            self.log_target).fit(np.c_[X, Zhat], y)
        self.G_ = make_G().fit(X, Z)
        self.Z_train_, self.Zhat_train_ = Z, Zhat
        return self

    def latent(self, X):
        return self.G_.predict(X)

    def predict(self, X):
        return self.H_.predict(np.c_[X, self.G_.predict(X)])


def transfer_fit_predict(student: str, p: dict, encoder, X_tr, y_tr, S_tr, xy_tr, X_te,
                         S_te=None, shuffle=False, true_latent=False, log_target=True,
                         seed=C.SEED, return_latent=False):
    if true_latent:
        rng = np.random.default_rng(seed)
        S_use = S_tr[rng.permutation(len(S_tr))] if shuffle else S_tr
        enc = encoder.fit(S_use)
        Z = enc.transform(S_use)
        H = LogTarget(make_student(student, p, seed), log_target).fit(np.c_[X_tr, Z], y_tr)
        Z_te = enc.transform(S_te)
        pred = H.predict(np.c_[X_te, Z_te])
        return (pred, Z_te, Z_te) if return_latent else pred
    m = TransferModel(student, p, encoder, log_target, shuffle, seed).fit(X_tr, y_tr, S_tr, xy_tr)
    pred = m.predict(X_te)
    if return_latent:
        Z_te_true = m.enc_.transform(S_te) if S_te is not None else None
        return pred, m.latent(X_te), Z_te_true
    return pred


def trend_fit_predict(xy_tr, y_tr, xy_te, alpha, log_target=True):
    est = Pipeline([("poly", PolynomialFeatures(2, include_bias=False)),
                    ("sc", StandardScaler()), ("m", Ridge(alpha=alpha))])
    return LogTarget(est, log_target).fit(xy_tr, y_tr).predict(xy_te)


def kriging_fit_predict(xy_tr, y_tr, xy_te, model="spherical"):
    from pykrige.ok import OrdinaryKriging
    ly = np.log(y_tr)
    ok = OrdinaryKriging(xy_tr[:, 0], xy_tr[:, 1], ly, variogram_model=model, nlags=12,
                         enable_plotting=False, verbose=False)
    z, ss = ok.execute("points", xy_te[:, 0], xy_te[:, 1])
    return np.exp(np.asarray(z) + 0.5 * np.asarray(ss))
