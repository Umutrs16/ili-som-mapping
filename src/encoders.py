"""Source-domain spectral encoders E fitted on public OSSL VisNIR spectra only."""
from __future__ import annotations

import pickle

import numpy as np
from sklearn.decomposition import PCA

from spectra import preprocess
from utils import C, get_logger

log = get_logger("encoders")
SOURCES = ("ICRAF-ISRIC", "LUCAS", "Pooled")
MAX_DIM = max(C.LATENT_DIMS)
ENC_PATH = C.MODELS / "public_encoders.pkl"


def load_library():
    z = np.load(C.OSSL_DIR / "ossl_visnir_topsoil_grid5nm.npz", allow_pickle=False)
    return z["refl"].astype(float), z["source"]


def source_rows(src: np.ndarray, name: str, rng: np.random.Generator) -> np.ndarray:
    icraf = np.where(src == "ICRAF-ISRIC")[0]
    lucas = np.where(src == "LUCAS")[0]
    if name == "ICRAF-ISRIC":
        return icraf
    if name == "LUCAS":
        return lucas
    # balanced pool: all ICRAF-ISRIC plus an equal-sized random LUCAS subset
    return np.r_[icraf, rng.choice(lucas, size=len(icraf), replace=False)]


def fit_public_encoders() -> dict:
    refl, src = load_library()
    rng = np.random.default_rng(C.SEED)
    rows = {s: source_rows(src, s, rng) for s in SOURCES}
    enc = {}
    for s in SOURCES:
        for prep in ("raw", "snv", "sg1d"):
            for dw in (False, True):
                x = preprocess(refl[rows[s]], prep, drop_water=dw)
                pca = PCA(MAX_DIM, random_state=C.SEED).fit(x)
                enc[(s, prep, dw)] = pca
                log.info("encoder %-11s %-4s water_dropped=%s n=%d EVR(8)=%.3f", s, prep, dw,
                         len(rows[s]), pca.explained_variance_ratio_.sum())
    with open(ENC_PATH, "wb") as f:
        pickle.dump({"encoders": enc, "n_rows": {s: len(r) for s, r in rows.items()}}, f)
    return enc


def load_public_encoders() -> dict:
    if not ENC_PATH.exists():
        fit_public_encoders()
    with open(ENC_PATH, "rb") as f:
        return pickle.load(f)["encoders"]


if __name__ == "__main__":
    fit_public_encoders()
