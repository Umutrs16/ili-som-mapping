"""All 20 test–calibration assignments (each held-out group × each other group).

    python s06b_rotation20.py
"""
from __future__ import annotations

import pandas as pd

import engine as En
from s05_experiments import E
from s06_uncertainty import conformal_rotation
from utils import C, get_logger

log = get_logger("s06b")


def experiment(name, student="ridge"):
    from s06_uncertainty import experiment as ex
    return ex(name, student)


def main():
    frames = []
    for name in ("E1", "E3"):
        exp = experiment(name)
        log.info(f"20 rotations: {name}")
        pts, summ = conformal_rotation(exp, all_calibrations=True)
        g = pts.groupby(["test_group", "cal_group"], as_index=False).agg(
            n_cov_abs=("cov_abs", "sum"), n_cov_log=("cov_log", "sum"),
            is_abs=("is_abs", "mean"), is_log=("is_log", "mean"))
        summ = summ.merge(g, on=["test_group", "cal_group"])
        summ.insert(0, "model", f"{name}__ridge")
        frames.append(summ)
        log.info(summ.groupby("test_group").cov_abs_cfg.agg(["min", "mean", "max"]).to_string())
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(C.RES / "conformal_rotation20.csv", index=False)
    log.info(f"wrote conformal_rotation20.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
