"""Trace files requested by the second review: selection log, map-fit
stability, conformal counts, laboratory paired differences, and block
intervals for the DI correlations.

    python s05f_r2_exports.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import engine as En
from s04_folds import blocks
from utils import C, get_logger, load_json, save_json

log = get_logger("s05f")
F = C.OM_REPORT_FACTOR


def selection_log():
    ns = load_json(C.RES / "nested_selection.json")
    rows = []
    for fold in ns["folds"]:
        for step in fold["steps"]["steps"]:
            for chk in step["checks"]:
                rows.append({"fold": fold["fold"], "held_out_group": f"G{fold['fold'] + 1}",
                             "candidate": step["candidate"], "adopted_before": step["current"],
                             "reference": chk["reference"],
                             "candidate_rmse_g_kg": fold["scores"][step["candidate"]] * F,
                             "reference_rmse_g_kg": fold["scores"][chk["reference"]] * F,
                             "delta_pct": chk["delta_pct"],
                             "middle_groups_improved": chk["groups_improved"],
                             "passes": chk["passes"], "step_passes": step["passes"],
                             "final_adopted": fold["adopted"], "final_student": fold["student"],
                             "theta": str(fold["theta"])})
        # candidates that were scored but not eligible after an early stop
        tested = {s["candidate"] for s in fold["steps"]["steps"]}
        for name, rmse in fold["scores"].items():
            if name not in tested:
                rows.append({"fold": fold["fold"], "held_out_group": f"G{fold['fold'] + 1}",
                             "candidate": name, "adopted_before": None, "reference": None,
                             "candidate_rmse_g_kg": rmse * F, "reference_rmse_g_kg": None,
                             "delta_pct": None, "middle_groups_improved": None,
                             "passes": None, "step_passes": None,
                             "final_adopted": fold["adopted"], "final_student": fold["student"],
                             "theta": str(fold["theta"]),
                             "note": "scored, not eligible after the sequence stopped"})
    pd.DataFrame(rows).to_csv(C.RES / "nested_selection_log.csv", index=False)
    log.info(f"selection log: {len(rows)} rows")


def stability():
    P = pd.read_csv(C.MAPS / "lattice_predictions.csv")
    m = P.valid.to_numpy()
    d = (P.OM_all - P.OM).to_numpy() * F
    d = d[m]
    flags = P["flags"].to_numpy()[m]
    om, om_all = P.OM.to_numpy()[m], P.OM_all.to_numpy()[m]
    inside = (flags & 1) == 0

    def cls(om_):
        return np.where(flags > 0, "B", np.where(om_ < 1.5, "A", "C"))

    a, b = cls(om), cls(om_all)
    qs = np.percentile(d, [5, 25, 50, 75, 95])
    out = {"n": int(m.sum()),
           "diff_mean_inside": float(d[inside].mean()),
           "diff_mae_inside": float(np.abs(d[inside]).mean()),
           "diff_mae_outside": float(np.abs(d[~inside]).mean()),
           "q05": float(qs[0]), "q25": float(qs[1]), "q50": float(qs[2]),
           "q75": float(qs[3]), "q95": float(qs[4]),
           "frac_abs_gt8": float(np.mean(np.abs(d) > 8)),
           "frac_below_colour5": float(np.mean(om * F < 5)),
           "frac_above_colour50": float(np.mean(om * F > 50)),
           "n_AC": int(np.isin(a, ["A", "C"]).sum()),
           "ac_changed": float(np.mean((a != b) & np.isin(a, ["A", "C"]))),
           "ac_changed_denominator": "class A or C transitions divided by all predictable nodes",
           "ac_changed_within_AC": float(((a != b) & np.isin(a, ["A", "C"])).sum()
                                         / np.isin(a, ["A", "C"]).sum()),
           "ac_changed_within_AC_denominator":
               "class A or C transitions divided by nodes that were class A or C before refitting",
           "a_to_c": int(((a == "A") & (b == "C")).sum()),
           "c_to_a": int(((a == "C") & (b == "A")).sum()),
           "n_A": int((a == "A").sum()), "n_C": int((a == "C").sum())}
    save_json(out, C.RES / "map_fit_stability.json")
    log.info(f"stability: inside MAE {out['diff_mae_inside']:.2f}; "
             f"A→C {out['a_to_c']}/{out['n_A']}; C→A {out['c_to_a']}/{out['n_C']}; "
             f"|diff|>8: {out['frac_abs_gt8']:.3f}")


def conformal_counts():
    cp = pd.read_csv(C.RES / "conformal_points.csv")
    out = {}
    for model, key in (("E1__ridge", "E1"), ("E3__ridge", "E3")):
        q = cp[cp.model == model]
        g5 = q[q.test_group == 4]
        out[key] = {"n": int(len(q)), "k_abs": int(q.cov_abs.sum()), "k_log": int(q.cov_log.sum()),
                    "g5_n": int(len(g5)), "g5_k_abs": int(g5.cov_abs.sum()),
                    "g5_k_log": int(g5.cov_log.sum())}
    save_json(out, C.RES / "conformal_counts.json")
    log.info(str(out))


def lab_pairs():
    D = En.get_data()
    bid = blocks(D.xy, 20)
    rows = []
    for learner, a, b in (("ridge", "E7b__ridge", "E7a__ridge"),
                          ("pls", "E7b__pls", "E7a")):
        ra, rb = En.load_run(a), En.load_run(b)
        bb = En.block_bootstrap_delta(D.y, ra["pred"], rb["pred"], bid)
        rows.append({"learner": learner, "model": "full spectrum", "reference": "simulated S2", **bb,
                     "rmse_full": rb["metrics"]["rmse"], "rmse_s2": ra["metrics"]["rmse"],
                     "r2_full": rb["metrics"]["r2"], "r2_s2": ra["metrics"]["r2"]})
    pd.DataFrame(rows).to_csv(C.RES / "lab_representation_pairs.csv", index=False)
    log.info(pd.DataFrame(rows)[["learner", "delta_pct", "pct_ci_low", "pct_ci_high"]].to_string(index=False))


def h4_block_rho():
    """Block-bootstrap interval for Spearman(DI, |error|). Ordinary p-values ignore space."""
    D = En.get_data()
    bid = blocks(D.xy, 20)
    rng = np.random.default_rng(C.SEED)
    ub = np.unique(bid)
    cp = pd.read_csv(C.RES / "conformal_points.csv")
    out = {}
    for model in ("E1__ridge", "E2__ridge", "E3__ridge"):
        q = cp[cp.model == model]
        # conformal_points may not be in sample order; match on y via the run
        run = En.load_run(model)
        err = np.abs(D.y - run["pred"])
        # DI from the outer-fold AOA is stored per point in sample order of the rotation file
        di = q.DI.to_numpy()
        # The rotation file repeats samples once in the primary rotation (one row per sample).
        if len(q) != len(D.y):
            raise SystemExit(f"{model} conformal rows {len(q)} != {len(D.y)}")
        # Align by the point order used when the file was written: test-group loop.
        # Safer: use the run's own nothing. Recompute rank correlation on the stored y.
        y = q.y.to_numpy()
        pred = q.pred.to_numpy()
        err = np.abs(y - pred)
        # block ids are not in this file. Join through sample order of outer groups.
        order = []
        for t in range(5):
            order.extend(np.where(D.groups == t)[0].tolist())
        if len(order) != len(q):
            raise SystemExit("cannot align conformal points with blocks")
        b = bid[np.array(order)]
        obs = float(spearmanr(di, err).statistic)
        boots = []
        for _ in range(C.N_BOOT):
            pick = np.concatenate([np.where(b == g)[0] for g in rng.choice(ub, len(ub))])
            boots.append(float(spearmanr(di[pick], err[pick]).statistic))
        out[model] = {"rho": float(obs),
                      "ci": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]}
    save_json(out, C.RES / "h4_block_spearman.json")
    log.info(str(out))


def main():
    selection_log()
    stability()
    conformal_counts()
    lab_pairs()
    h4_block_rho()


if __name__ == "__main__":
    main()
