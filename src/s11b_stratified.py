"""High-SOM and spatial-group stratified error diagnostics (review item F).

    python s11b_stratified.py
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from utils import C, get_logger, save_json

log = get_logger("s11b")
F = C.OM_REPORT_FACTOR


def metrics(y, p):
    e = p - y
    return {"n": int(len(y)), "bias": float(e.mean() * F), "mae": float(np.abs(e).mean() * F),
            "rmse": float(np.sqrt(np.mean(e ** 2)) * F),
            "slope": float(np.polyfit(y, p, 1)[0]) if len(y) > 1 else None}


def main():
    soil = pd.read_csv(C.RAW / "soil_audited.csv", index_col="sample_id")
    folds = pd.read_csv(C.FOLDS / "folds.csv", index_col="sample_id")
    y = soil.OM.to_numpy(float)
    # fixed terciles from the full sample (documented; not tuned on test)
    t1, t2 = np.percentile(y, [100 / 3, 200 / 3])
    rows = []
    for tag, label in (("E1__ridge", "E1 ridge"), ("E3__ridge", "E3 ridge"),
                       ("nested_selection", "nested selection")):
        path = C.OOF / f"{tag}.pkl"
        if not path.exists():
            continue
        with open(path, "rb") as f:
            r = pickle.load(f)
        pred = r["pred"]
        m = ~np.isnan(pred)
        for name, mask in (
            ("all", m),
            ("SOM_low", m & (y <= t1)),
            ("SOM_mid", m & (y > t1) & (y <= t2)),
            ("SOM_high", m & (y > t2)),
            *[(f"G{g + 1}", m & (folds.outer_group.to_numpy() == g)) for g in range(5)],
        ):
            rows.append({"model": label, "stratum": name, "thr_low": float(t1 * F),
                         "thr_high": float(t2 * F), **metrics(y[mask], pred[mask])})
    df = pd.DataFrame(rows)
    df.to_csv(C.RES / "stratified_errors.csv", index=False)
    save_json({"tercile_g_kg": [float(t1 * F), float(t2 * F)], "n_rows": len(df)},
              C.RES / "stratified_errors_meta.json")
    log.info(f"wrote stratified_errors.csv ({len(df)} rows); terciles "
             f"{t1 * F:.1f} / {t2 * F:.1f} g kg-1")
    print(df[df.stratum.isin(["all", "SOM_low", "SOM_high"])].to_string(index=False))


if __name__ == "__main__":
    main()
