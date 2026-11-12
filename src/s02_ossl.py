"""Step 2 - Open Soil Spectral Library (OSSL v1.2) VisNIR source domains.

Downloads (if missing) the OSSL VisNIR, site and laboratory tables, keeps the
ICRAF-ISRIC and LUCAS libraries (LUCAS.WOODWELL duplicates LUCAS soils and is
excluded), retains topsoil layers with complete 400-2450 nm support, merges
replicate scans per layer and resamples everything onto the common 5 nm grid.
"""
from __future__ import annotations

import urllib.request

import numpy as np
import pandas as pd

from spectra import GRID, smooth_to_grid
from utils import C, get_logger, save_json, sha256

log = get_logger("s02_ossl")
SOURCES = {"ICRAF.ISRIC": "ICRAF-ISRIC", "LUCAS.SSL": "LUCAS"}
WL_OSSL = np.arange(350, 2501, 2)
SPEC_COLS = [f"scan_visnir.{w}_ref" for w in WL_OSSL]
META = ["dataset.code_ascii_txt", "id.layer_uuid_txt", "id.scan_local_c",
        "scan.visnir.model.name_utf8_txt", "scan.visnir.method.preparation_any_txt",
        "scan.visnir.license.title_ascii_txt", "scan.visnir.doi_idf_url"]
OC_COLS = ["oc_usda.c729_w.pct", "oc_usda.c1059_w.pct", "oc_iso.10694_w.pct",
           "c.tot_usda.a622_w.pct", "c.tot_iso.10694_w.pct", "caco3_usda.a54_w.pct",
           "caco3_iso.10693_w.pct", "clay.tot_usda.a334_w.pct", "clay.tot_iso.11277_w.pct",
           "ph.h2o_usda.a268_index", "ph.h2o_iso.10390_index"]


def download() -> dict:
    info = {}
    for key, fn in C.OSSL_FILES.items():
        p = C.OSSL_DIR / fn
        if not p.exists():
            log.info("downloading %s", fn)
            urllib.request.urlretrieve(C.OSSL_URL + fn, p)
        info[fn] = {"url": C.OSSL_URL + fn, "bytes": p.stat().st_size, "sha256": sha256(p)}
    return info


def main() -> None:
    files = download()
    keep_codes = set(SOURCES)
    parts = []
    for chunk in pd.read_csv(C.OSSL_DIR / C.OSSL_FILES["visnir"], usecols=META + SPEC_COLS,
                             chunksize=10000, low_memory=False):
        parts.append(chunk[chunk["dataset.code_ascii_txt"].isin(keep_codes)])
    vis = pd.concat(parts, ignore_index=True)
    n_raw = vis.groupby("dataset.code_ascii_txt").size().to_dict()

    site = pd.read_csv(C.OSSL_DIR / C.OSSL_FILES["site"], low_memory=False,
                       usecols=["id.layer_uuid_txt", "layer.upper.depth_usda_cm",
                                "layer.lower.depth_usda_cm", "location.country_iso.3166_txt",
                                "longitude.point_wgs84_dd", "latitude.point_wgs84_dd"])
    lab_cols = pd.read_csv(C.OSSL_DIR / C.OSSL_FILES["lab"], nrows=0).columns
    lab = pd.read_csv(C.OSSL_DIR / C.OSSL_FILES["lab"], low_memory=False,
                      usecols=["id.layer_uuid_txt"] + [c for c in OC_COLS if c in lab_cols])
    vis = vis.merge(site, on="id.layer_uuid_txt", how="left") \
             .merge(lab, on="id.layer_uuid_txt", how="left")

    grid_cols = [f"scan_visnir.{w}_ref" for w in WL_OSSL if 400 <= w <= 2450]
    complete = vis[grid_cols].notna().all(axis=1)
    topsoil = vis["layer.upper.depth_usda_cm"].fillna(0) < 20
    steps = {"complete_400_2450": complete.groupby(vis["dataset.code_ascii_txt"]).sum().to_dict(),
             "topsoil_and_complete": (complete & topsoil).groupby(
                 vis["dataset.code_ascii_txt"]).sum().to_dict()}
    vis = vis[complete & topsoil].copy()

    # merge replicate scans of the same layer
    dup = vis["id.layer_uuid_txt"].duplicated(keep=False)
    steps["layers_with_replicates"] = int(vis.loc[dup, "id.layer_uuid_txt"].nunique())
    num_cols = SPEC_COLS + [c for c in OC_COLS if c in vis.columns] + \
        ["layer.upper.depth_usda_cm", "layer.lower.depth_usda_cm",
         "longitude.point_wgs84_dd", "latitude.point_wgs84_dd"]
    first_cols = ["dataset.code_ascii_txt", "location.country_iso.3166_txt",
                  "scan.visnir.model.name_utf8_txt", "scan.visnir.license.title_ascii_txt"]
    agg = vis.groupby("id.layer_uuid_txt").agg({**{c: "mean" for c in num_cols},
                                                **{c: "first" for c in first_cols}})

    refl = agg[SPEC_COLS].to_numpy(float)
    # fill isolated gaps outside 400-2450 by edge extension before smoothing
    refl = pd.DataFrame(refl).interpolate(axis=1, limit_direction="both").to_numpy()
    grid = smooth_to_grid(WL_OSSL.astype(float), refl, window_nm=21.0)

    def first_valid(cols):
        cols = [c for c in cols if c in agg.columns]
        return agg[cols].bfill(axis=1).iloc[:, 0].to_numpy(float) if cols else np.full(len(agg), np.nan)

    oc = first_valid(["oc_usda.c729_w.pct", "oc_usda.c1059_w.pct", "oc_iso.10694_w.pct"])
    caco3 = first_valid(["caco3_usda.a54_w.pct", "caco3_iso.10693_w.pct"])
    clay = first_valid(["clay.tot_usda.a334_w.pct", "clay.tot_iso.11277_w.pct"])
    ph = first_valid(["ph.h2o_usda.a268_index", "ph.h2o_iso.10390_index"])
    src = agg["dataset.code_ascii_txt"].map(SOURCES).to_numpy()

    np.savez_compressed(C.OSSL_DIR / "ossl_visnir_topsoil_grid5nm.npz",
                        grid=GRID, refl=grid.astype(np.float32),
                        uuid=agg.index.to_numpy().astype(str), source=src.astype(str),
                        oc=oc, caco3=caco3, clay=clay, ph=ph,
                        country=agg["location.country_iso.3166_txt"].astype(str).to_numpy(),
                        upper=agg["layer.upper.depth_usda_cm"].to_numpy(float),
                        lower=agg["layer.lower.depth_usda_cm"].to_numpy(float))
    summary = {
        "ossl_version": C.OSSL_VERSION, "files": files,
        "datasets_kept": SOURCES, "excluded_note":
            "LUCAS.WOODWELL.SSL excluded (re-scanned subset of LUCAS soils); KSSL not used",
        "n_raw_records": n_raw, "filter_steps": steps,
        "n_final": pd.Series(src).value_counts().to_dict(),
        "instrument": agg.groupby("dataset.code_ascii_txt")["scan.visnir.model.name_utf8_txt"]
                         .first().to_dict(),
        "license": agg.groupby("dataset.code_ascii_txt")["scan.visnir.license.title_ascii_txt"]
                      .first().to_dict(),
        "oc_pct_summary": {s: pd.Series(oc[src == s]).describe().to_dict()
                           for s in np.unique(src)},
        "countries": {s: int(pd.Series(agg["location.country_iso.3166_txt"].to_numpy()[src == s])
                             .nunique()) for s in np.unique(src)},
    }
    save_json(summary, C.OSSL_DIR / "ossl_summary.json")
    log.info("OSSL harmonised: %s", summary["n_final"])


if __name__ == "__main__":
    main()
