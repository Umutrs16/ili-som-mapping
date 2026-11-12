"""Step 0 - data audit, data dictionary and sample manifest.

Reproduces the checks of the study framework (file structure, ID linkage,
duplicates, missing values, spatial extent, nearest-neighbour spacing, CEC
consistency, zero codes) and freezes an audited copy of the inputs.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from utils import C, get_logger, load_soil, load_spectra, save_json, sha256, utm_xy

log = get_logger("s00_audit")

UNITS = {
    "soil_moisture": ("Soil moisture", "as recorded (fraction)"),
    "NO3_N": ("Nitrate nitrogen", "as recorded"),
    "NH4_N": ("Ammonium nitrogen", "as recorded"),
    "P": ("Phosphorus (fraction not documented)", "as recorded"),
    "K": ("Potassium (fraction not documented)", "as recorded"),
    "Ca": ("Calcium", "as recorded"), "Mg": ("Magnesium", "as recorded"),
    "S": ("Sulphur", "as recorded"), "Fe": ("Iron", "as recorded"),
    "Mn": ("Manganese", "as recorded"), "B": ("Boron", "as recorded"),
    "Cu": ("Copper", "as recorded"), "Zn": ("Zinc", "as recorded"),
    "Mo": ("Molybdenum", "as recorded"), "Na": ("Sodium", "as recorded"),
    "Al": ("Aluminium", "as recorded"), "Cl": ("Chloride", "as recorded"),
    "pH": ("Soil pH", "unitless (extractant not documented)"),
    "EC": ("Electrical conductivity", "as recorded (soil:water ratio not documented)"),
    "CEC": ("Cation exchange capacity", "as recorded"),
    "OM": ("Recorded organic matter; dichromate organic carbon, no conversion factor",
           "% (g per 100 g); reported SOM (g kg-1) = 10 x this value"),
}


def describe(s: pd.Series) -> dict:
    v = s.dropna().to_numpy(float)
    return {
        "n": int(v.size), "min": v.min(), "max": v.max(), "mean": v.mean(),
        "median": np.median(v), "sd": v.std(ddof=1),
        "cv_pct": 100 * v.std(ddof=1) / v.mean() if v.mean() != 0 else np.nan,
        "skewness": stats.skew(v, bias=False), "q25": np.percentile(v, 25),
        "q75": np.percentile(v, 75),
    }


def main() -> None:
    soil = load_soil()
    wl, spec = load_spectra()
    numeric = [c for c in soil.columns if c not in
               ("field_id", "lon", "lat", "date", "depth", "land_use",
                "sampling_method", "analysis_method")]

    audit: dict = {"run_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
    audit["soil_shape"] = [int(soil.shape[0]), int(soil.shape[1] + 1)]
    audit["numeric_soil_variables"] = numeric
    audit["n_numeric_soil_variables"] = len(numeric)
    audit["spectra_shape"] = [int(spec.shape[1]), int(spec.shape[0])]
    audit["wavelength_range_nm"] = [float(wl.min()), float(wl.max())]
    audit["wavelength_step_nm"] = sorted(set(np.diff(wl).round(6).tolist()))

    ids_soil, ids_spec = set(soil.index), set(spec.index)
    audit["ids_matched"] = len(ids_soil & ids_spec)
    audit["ids_only_soil"] = sorted(ids_soil - ids_spec)
    audit["ids_only_spectra"] = sorted(ids_spec - ids_soil)
    spec = spec.loc[soil.index]

    audit["duplicate_ids"] = int(soil.index.duplicated().sum())
    audit["duplicate_coordinates"] = int(soil.duplicated(subset=["lon", "lat"]).sum())
    audit["spectra_missing_values"] = int(spec.isna().sum().sum())
    audit["identical_spectra_pairs"] = int(spec.round(6).duplicated().sum())
    audit["unique_dates"] = sorted({str(pd.Timestamp(d).date()) for d in soil["date"]})
    audit["unique_land_use"] = sorted(soil["land_use"].astype(str).unique())
    audit["unique_depth"] = sorted(soil["depth"].astype(str).unique())
    audit["unique_sampling_method"] = sorted(soil["sampling_method"].astype(str).unique())
    audit["analysis_method_all_missing"] = bool(soil["analysis_method"].isna().all())
    audit["field_id_unique"] = bool(soil["field_id"].is_unique)
    audit["lon_range"] = [soil["lon"].min(), soil["lon"].max()]
    audit["lat_range"] = [soil["lat"].min(), soil["lat"].max()]
    miss = soil[numeric].isna()
    audit["missing_by_variable"] = {c: soil.index[miss[c]].tolist()
                                    for c in numeric if miss[c].any()}

    x, y = utm_xy(soil["lon"], soil["lat"])
    d, _ = cKDTree(np.c_[x, y]).query(np.c_[x, y], k=2)
    nn = d[:, 1]
    audit["nn_distance_m"] = {"median": np.median(nn), "min": nn.min(),
                              "max": nn.max(), "mean": nn.mean()}
    audit["n_pairs_within_500m"] = int((nn < 500).sum())

    # CEC versus base cations converted from mg/kg to cmol(+)/kg
    eq = soil["Ca"] / 200.4 + soil["Mg"] / 121.5 + soil["K"] / 391.0 + soil["Na"] / 230.0
    r_cec = np.corrcoef(eq, soil["CEC"])[0, 1]
    lr = stats.linregress(eq, soil["CEC"])
    audit["cec_vs_base_cations"] = {"pearson_r": r_cec, "slope": lr.slope,
                                    "intercept": lr.intercept}
    audit["mo_zero_count"] = int((soil["Mo"] == 0).sum())
    audit["reflectance_range"] = [float(spec.values.min()), float(spec.values.max())]
    audit["om_range"] = [soil["OM"].min(), soil["OM"].max()]

    stats_tbl = pd.DataFrame({c: describe(soil[c]) for c in numeric}).T
    stats_tbl.index.name = "variable"
    stats_tbl.to_csv(C.AUDIT / "descriptive_statistics.csv", float_format="%.6g")

    rows = []
    for col in ["sample_id"] + list(soil.columns):
        if col == "sample_id":
            rows.append(dict(field=col, description="Sample identifier (join key to spectra)",
                             unit="-", missing=0))
            continue
        name, unit = UNITS.get(col, (col, "-"))
        if col == "lon":
            name, unit = "Longitude", "degree (WGS84 assumed)"
        elif col == "lat":
            name, unit = "Latitude", "degree (WGS84 assumed)"
        elif col == "depth":
            name, unit = "Sampling depth", "0-20 cm composite (confirmed)"
        elif col == "date":
            name, unit = "Sampling date", "YYYY-MM-DD"
        rows.append(dict(field=col, description=name, unit=unit,
                         missing=int(soil[col].isna().sum())))
    pd.DataFrame(rows).to_csv(C.AUDIT / "data_dictionary.csv", index=False)

    manifest = {
        "inputs": {p.name: {"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
                   for p in (C.SOIL_XLSX, C.SPEC_XLSX, C.ROI_SHP,
                             C.ROI_SHP.with_suffix(".dbf"), C.ROI_SHP.with_suffix(".shx"),
                             C.ROI_SHP.with_suffix(".prj"))},
        "join_key": "sample_id",
        "n_samples": int(soil.shape[0]),
    }
    save_json(manifest, C.AUDIT / "sample_manifest.json")

    out = soil.copy()
    out["x_utm"], out["y_utm"] = x, y
    out["nn_dist_m"] = nn
    out.to_csv(C.RAW / "soil_audited.csv")
    spec.to_csv(C.RAW / "spectra_audited.csv", float_format="%.5f")
    save_json(audit, C.AUDIT / "audit_report.json")
    log.info("Audit done: %d samples, %d numeric vars, NN median %.0f m, min %.0f m",
             soil.shape[0], len(numeric), np.median(nn), nn.min())
    log.info("CEC~base cations r = %.7f; Mo zeros = %d", r_cec, audit["mo_zero_count"])


if __name__ == "__main__":
    main()
