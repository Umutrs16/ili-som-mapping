"""Adoption rule inside the 86 mapping-fit samples only.

The 28 calibration samples are not used for tuning or for the adoption decision.
They remain reserved for the conformal quantile.

    python s05e_fitset.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine as En
from s05c_nested_select import CANDIDATES, adopt_in_train, middle_score
from utils import C, get_logger, save_json

log = get_logger("s05e")


def main():
    D = En.get_data()
    split = pd.read_csv(C.FOLDS / "deployment_split.csv")
    fit = np.where(split.role.to_numpy() == "fit")[0]
    cal = np.where(split.role.to_numpy() == "cal")[0]
    log.info(f"fit-set adoption: n_fit={len(fit)} n_cal={len(cal)} (calibration unused)")
    scored = {}
    for name, exp in CANDIDATES.items():
        rmse, theta, fold_rmse = middle_score(exp, fit)
        scored[name] = {"rmse": rmse, "theta": theta, "fold_rmse": fold_rmse}
        log.info(f"  {name}: middle RMSE={rmse * C.OM_REPORT_FACTOR:.2f}")
    adopted, student, steps = adopt_in_train(fit, scored)
    out = {"n_fit": int(len(fit)), "n_cal": int(len(cal)),
           "adopted": adopted, "student": student,
           "theta": scored[adopted if student == "ridge" else f"{adopted}__{student}"]["theta"],
           "scores_g_kg": {k: v["rmse"] * C.OM_REPORT_FACTOR for k, v in scored.items()},
           "steps": steps}
    save_json(out, C.RES / "fitset_selection.json")
    log.info(f"fit-set selection: {adopted} / {student} theta={out['theta']}")


if __name__ == "__main__":
    main()
