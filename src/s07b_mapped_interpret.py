"""Mapped-model interpretation and the remaining second-review traces.

Archives the engineering-sequence SHAP tables, appends grouped permutation
importance for E1, writes E1 SHAP from the mapped linear model, and recomputes
block-bootstrap intervals for Spearman(DI, |error|) from the outer-fold AOA
tables.

    python s07b_mapped_interpret.py
"""
from __future__ import annotations

import hashlib
import shutil

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import engine as En
from s05_experiments import tag_of
from s06_uncertainty import experiment
from s07_interpret import grouped_permutation, linear_shap
from utils import C, get_logger, load_json, save_json

log = get_logger("s07b")


def archive_shap():
    for name in ("shap_feature_summary.csv", "shap_linear_values.csv", "shap_grouped_values.csv"):
        src, dst = C.RES / name, C.RES / name.replace(".csv", "_E3.csv")
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
            log.info(f"archived {dst.name}")


def e1_importance():
    exp = experiment("E1", "ridge")
    run = En.load_run(tag_of(exp))
    df, rmse0, _ = grouped_permutation(exp, run)
    df.insert(0, "model", "E1__ridge")
    df["rmse_base"] = rmse0
    path = C.RES / "grouped_permutation_importance.csv"
    old = pd.read_csv(path)
    old = old[old.model != "E1__ridge"]
    pd.concat([df, old], ignore_index=True).to_csv(path, index=False)
    log.info("E1 grouped permutation:\n" + df.to_string(index=False))
    archive_shap()
    linear_shap()
    shutil.copy(C.RES / "shap_feature_summary.csv", C.RES / "shap_feature_summary_E1.csv")


def h4_block():
    import engine as En
    D = En.get_data()
    bid = pd.Series(D.blocks, index=D.ids)
    rng = np.random.default_rng(C.SEED + 11)
    out = {}
    for nm in ("E1", "E2", "E3"):
        df = pd.read_csv(C.RES / f"oof_aoa_{nm}.csv")
        di, err = df.DI.to_numpy(), df.abs_err.to_numpy()
        b = bid.loc[df.sample_id].to_numpy()
        ub = np.unique(b)
        obs = float(spearmanr(di, err).statistic)
        boots = np.empty(C.N_BOOT)
        for i in range(C.N_BOOT):
            pick = np.concatenate([np.where(b == g)[0] for g in rng.choice(ub, len(ub))])
            boots[i] = float(spearmanr(di[pick], err[pick]).statistic)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        out[nm] = {"rho": obs, "ci": [float(lo), float(hi)]}
        log.info(f"H4 {nm}: rho={obs:.3f} block CI {lo:.3f} to {hi:.3f}")
    save_json(out, C.RES / "h4_block_spearman.json")


def extras():
    st = load_json(C.RES / "map_fit_stability.json")
    P = pd.read_csv(C.MAPS / "lattice_predictions.csv")
    d = np.abs((P.OM_all - P.OM).to_numpy() * C.OM_REPORT_FACTOR)
    d = d[P.valid.to_numpy()]
    st["frac_abs_gt15"] = float(np.mean(d > 15))
    save_json(st, C.RES / "map_fit_stability.json")
    digest = hashlib.sha256(C.TABPFN_CKPT.read_bytes()).hexdigest()
    (C.RES / "tabpfn_sha256.txt").write_text(digest, encoding="utf-8")
    log.info(f"|diff|>15: {st['frac_abs_gt15']:.3f}; SHA-256 {digest}")
    prot = {
        "depth_cm": "0-20",
        "field": "five-point composite; bagged, numbered and sealed; sample-centre location recorded",
        "preparation": "visible impurities removed; air-dried indoors in a ventilated place; ground; 0.15 mm sieve",
        "carbon_method": "potassium dichromate volumetric method with external heating",
        "recorded_column": "OM, percent (g per 100 g), confirmed by the data provider as organic matter",
        "conversion_applied": None,
        "reporting": "SOM (g kg-1) = 10 x recorded percent",
        "spectrometer": "SR-8800 portable spectroradiometer",
        "spectra": "dark room; white-panel calibration before measurement; five scans; arithmetic mean; 350-2500 nm at 1 nm; reflectance in percent",
        "still_unconfirmed": ["pH method", "EC method and units", "whether moisture is gravimetric or volumetric"],
    }
    save_json(prot, C.AUDIT / "laboratory_protocol.json")
    dic = pd.read_csv(C.AUDIT / "data_dictionary.csv")
    dic.loc[dic.field == "OM", "description"] = (
        "Recorded organic matter. Organic carbon was measured by potassium dichromate "
        "with external heating; no conversion factor is applied")
    dic.loc[dic.field == "OM", "unit"] = "% (g per 100 g); reported SOM (g kg-1) = 10 x this value"
    dic.loc[dic.field == "depth", "unit"] = "0-20 cm composite (confirmed)"
    dic.to_csv(C.AUDIT / "data_dictionary.csv", index=False)


def main():
    e1_importance()
    h4_block()
    extras()


if __name__ == "__main__":
    main()
