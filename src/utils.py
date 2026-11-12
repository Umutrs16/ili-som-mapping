"""Shared helpers: logging, hashing, JSON I/O and the canonical data loaders."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

# a PROJ_LIB/PROJ_DATA without proj.db (set by other software) hides the PROJ data shipped with rasterio
for _k in ("PROJ_LIB", "PROJ_DATA"):
    if not (Path(os.environ.get(_k, "").strip('"')) / "proj.db").is_file():
        os.environ.pop(_k, None)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(C.LOGS / "run_log.txt", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=_NpEncoder)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def timer(logger: logging.Logger, label: str):
    t0 = time.time()
    yield
    logger.info("%s finished in %.1f s", label, time.time() - t0)


def load_soil() -> pd.DataFrame:
    """Soil attribute table indexed by sample_id (original units)."""
    df = pd.read_excel(C.SOIL_XLSX)
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    return df.set_index("sample_id").sort_index()


def load_spectra() -> tuple[np.ndarray, pd.DataFrame]:
    """Laboratory reflectance (as recorded, %) -> (wavelengths, samples x wavelengths)."""
    raw = pd.read_excel(C.SPEC_XLSX)
    wl = raw["Wavelength"].to_numpy(dtype=float)
    spec = raw.drop(columns=["Wavelength"]).T
    spec.index = spec.index.astype(str).str.strip()
    spec.columns = wl.astype(int)
    return wl, spec.sort_index()


def utm_xy(lon, lat):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", C.CRS_UTM, always_xy=True)
    x, y = tr.transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x), np.asarray(y)
