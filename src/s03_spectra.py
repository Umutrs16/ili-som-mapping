"""Step 3 - local laboratory spectra, PLS5 screening reproduction and SRF bridging.

* Smooths the 1 nm laboratory spectra (SG, 21 nm, 2nd order) onto the common
  400-2450 nm / 5 nm grid; reflectance recorded in percent is divided by 100.
* Reproduces the fixed-parameter PLS5 screening reported in the study plan.
* Convolves the laboratory spectra with the Sentinel-2 spectral response
  functions (ESA COPE-GSEG-EOPG-TN-15-0007; v4.0 = reference for the imagery
  used here, v5.0 = comparison) and weights S2A/S2B/S2C by the per-sample
  share of bare-soil observations from each spacecraft.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from spectra import GRID, smooth_to_grid
from utils import C, get_logger, load_soil, load_spectra, save_json

log = get_logger("s03_spectra")
S2_SIM_BANDS = C.S2_BANDS


def pls5_screen(grid_refl: np.ndarray, soil: pd.DataFrame) -> pd.DataFrame:
    """Fixed PLS (5 LV, scale=False): random 5-fold vs longitude-sorted 5 groups."""
    rows = []
    order = np.argsort(soil["lon"].to_numpy())
    groups = np.empty(len(soil), int)
    for g, idx in enumerate(np.array_split(order, 5)):
        groups[idx] = g
    for target in [C.TARGET] + C.AUX_TARGETS:
        y = soil[target].to_numpy(float)
        res = {}
        for scheme in ("random", "lon_groups"):
            pred = np.empty_like(y)
            base = np.empty_like(y)
            if scheme == "random":
                splits = KFold(5, shuffle=True, random_state=C.SEED).split(grid_refl)
            else:
                splits = ((np.where(groups != g)[0], np.where(groups == g)[0]) for g in range(5))
            for tr, te in splits:
                m = PLSRegression(n_components=5, scale=False).fit(grid_refl[tr], y[tr])
                pred[te] = m.predict(grid_refl[te]).ravel()
                base[te] = y[tr].mean()
            ss = ((y - y.mean()) ** 2).sum()
            res[scheme] = dict(r2=1 - ((y - pred) ** 2).sum() / ss,
                               rmse=np.sqrt(((y - pred) ** 2).mean()),
                               base_rmse=np.sqrt(((y - base) ** 2).mean()))
        rows.append(dict(target=target, r2_random=res["random"]["r2"],
                         r2_lon=res["lon_groups"]["r2"], rmse_lon=res["lon_groups"]["rmse"],
                         base_rmse_lon=res["lon_groups"]["base_rmse"]))
    return pd.DataFrame(rows)


def load_srf(version: str) -> dict[str, pd.DataFrame]:
    path = C.SRF_DIR / f"S2-SRF_v{version}.xlsx"
    out = {}
    for sc in ("S2A", "S2B", "S2C"):
        df = pd.read_excel(path, sheet_name=f"Spectral Responses ({sc})")
        df = df.rename(columns={"SR_WL": "wl"}).set_index("wl")
        df.columns = [c.replace(f"{sc}_SR_AV_", "") for c in df.columns]
        out[sc] = df[S2_SIM_BANDS].fillna(0.0).clip(lower=0.0)
    return out


def convolve(wl: np.ndarray, refl: np.ndarray, srf: pd.DataFrame) -> np.ndarray:
    """Eq. (4): R_b = sum R(l) S_b(l) dl / sum S_b(l) dl on the 1 nm laboratory grid."""
    s = srf.reindex(wl).fillna(0.0).to_numpy()
    return refl @ s / s.sum(axis=0, keepdims=True)


def main() -> None:
    soil = load_soil()
    wl, spec = load_spectra()
    refl_frac = spec.loc[soil.index].to_numpy(float) / 100.0
    grid = smooth_to_grid(wl, refl_frac, window_nm=21.0)
    np.savez_compressed(C.FEAT / "local_spectra_grid5nm.npz", grid=GRID, refl=grid,
                        sample_id=soil.index.to_numpy().astype(str))

    screen = pls5_screen(grid, soil)
    screen.to_csv(C.RES / "pls5_screening.csv", index=False)
    log.info("PLS5 screening:\n%s", screen.round(3).to_string(index=False))

    qa_path = C.GEE_DIR / "points_main_s20_buffer.csv"
    weights = None
    if qa_path.exists():
        qa = pd.read_csv(qa_path, index_col="sample_id").loc[soil.index]
        w = qa[["n_S2A", "n_S2B", "n_S2C"]].to_numpy(float)
        weights = w / np.clip(w.sum(axis=1, keepdims=True), 1e-9, None)
    sims = {}
    for ver in ("4.0", "5.0"):
        srf = load_srf(ver)
        per_sc = {sc: convolve(wl, refl_frac, srf[sc]) for sc in srf}
        if weights is None:
            comp = np.mean([per_sc[s] for s in per_sc], axis=0)
        else:
            comp = sum(weights[:, [k]] * per_sc[sc] for k, sc in enumerate(("S2A", "S2B", "S2C")))
        sims[ver] = pd.DataFrame(comp, index=soil.index, columns=[f"sim_{b}" for b in S2_SIM_BANDS])
        for sc, arr in per_sc.items():
            pd.DataFrame(arr, index=soil.index, columns=S2_SIM_BANDS) \
                .to_csv(C.FEAT / f"lab_sim_{sc}_v{ver}.csv", float_format="%.6f")
    sims["4.0"].to_csv(C.FEAT / "lab_sim_s2_weighted_v4.0.csv", float_format="%.6f")
    sims["5.0"].to_csv(C.FEAT / "lab_sim_s2_weighted_v5.0.csv", float_format="%.6f")
    diff = (sims["5.0"].to_numpy() - sims["4.0"].to_numpy())
    summary = {
        "weights_from_gee": weights is not None,
        "mean_spacecraft_share": None if weights is None else weights.mean(axis=0).tolist(),
        "srf_v5_minus_v4_abs_max": float(np.abs(diff).max()),
        "srf_v5_minus_v4_abs_mean_by_band": dict(zip(S2_SIM_BANDS, np.abs(diff).mean(axis=0))),
        "reflectance_scale_note": "laboratory reflectance recorded in percent; divided by 100",
    }
    save_json(summary, C.RES / "srf_summary.json")
    log.info("SRF simulation done; |v5-v4| max = %.5f", summary["srf_v5_minus_v4_abs_max"])


if __name__ == "__main__":
    main()
