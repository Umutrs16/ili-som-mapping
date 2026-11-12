"""Outer-fold nested adoption (review experiment A).

For each outer spatial fold, the adoption rule of s05b_compare is applied using
only middle-level spatial CV scores computed inside that fold's training set.
The selected (experiment, learner) is then fitted on the full outer training set
and used once to predict the held-out group. Pooled predictions are the
selection-procedure estimate; they must not be used again to choose a model.

    python s05c_nested_select.py
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import engine as En
import models as M
from s04_folds import middle_splits_kmeans, outer_splits
from s05b_compare import GAIN, MIN_GROUPS
from utils import C, get_logger, save_json

log = get_logger("s05c_nested")

# Full adoption sequence inside each outer training set. Learners other than ridge
# are compared only when the adopted experiment is a plain covariate model.
PUBLIC = "ICRAF-ISRIC"
_B = M.BASE_SETS


def _exp(name):
    spec = {
        "E0b": dict(kind="trend"),
        "E1": dict(kind="student", groups=_B["E1"]),
        "E2": dict(kind="student", groups=_B["E2"]),
        "E3": dict(kind="student", groups=_B["E3"]),
        "E4a": dict(kind="transfer", groups=_B["E2"], encoder="local"),
        "E4b": dict(kind="transfer", groups=_B["E3"], encoder="local"),
        "E5a": dict(kind="transfer", groups=_B["E2"], encoder=PUBLIC),
        "E5b": dict(kind="transfer", groups=_B["E3"], encoder=PUBLIC),
        "E6a-L": dict(kind="transfer", groups=_B["E2"], encoder="local", shuffle=True),
        "E6a-P": dict(kind="transfer", groups=_B["E2"], encoder=PUBLIC, shuffle=True),
        "E6b-L": dict(kind="transfer", groups=_B["E3"], encoder="local", shuffle=True),
        "E6b-P": dict(kind="transfer", groups=_B["E3"], encoder=PUBLIC, shuffle=True),
    }[name]
    return En.Experiment(name, student="ridge", **spec)


CANDIDATES = {n: _exp(n) for n in
              ["E0b", "E1", "E2", "E3", "E5a", "E5b", "E4a", "E4b",
               "E6a-L", "E6a-P", "E6b-L", "E6b-P"]}
NEST_ORDER = ["E1", "E2", "E3", "E5a", "E5b", "E4a", "E4b"]
NEST_REFS = {"E1": ["E0b"], "E2": ["E1"], "E3": ["E2"],
             "E4a": ["E2", "E6a-L"], "E5a": ["E2", "E6a-P"],
             "E4b": ["E3", "E6b-L"], "E5b": ["E3", "E6b-P"]}
LEARNERS = ("ridge", "pls")


def middle_score(exp: En.Experiment, tr: np.ndarray, dk=En.data_key(),
                 buffer_km: float = C.BUFFER_MAIN_KM):
    """Best middle-CV RMSE and its theta inside one outer training set."""
    D = En.get_data(*dk)
    theta = En.select_params(exp, [tr], dk, buffer_km, n_jobs=6)[0]
    mids = middle_splits_kmeans(tr, D.xy, buffer_km)
    sq, fold_rmse = [], []
    for _, tr_m, te_m in mids:
        pred = En.fit_predict(exp, theta, tr_m, te_m, dk)
        e = (D.y[te_m] - pred) ** 2
        sq.append(e)
        fold_rmse.append(float(np.sqrt(np.mean(e))))
    return float(np.sqrt(np.mean(np.concatenate(sq)))), theta, fold_rmse


def adopt_in_train(tr: np.ndarray, scored: dict) -> tuple[str, str, dict]:
    adopted, detail = None, []
    for m in NEST_ORDER:
        refs = list(NEST_REFS[m]) + ([adopted] if adopted and adopted not in NEST_REFS[m] else [])
        checks = []
        ok = True
        for r in refs:
            dlt = 100 * (scored[m]["rmse"] / scored[r]["rmse"] - 1)
            # middle-fold wins: compare per-fold RMSE vectors
            g = int(np.sum(np.asarray(scored[m]["fold_rmse"]) < np.asarray(scored[r]["fold_rmse"])))
            checks.append({"reference": r, "delta_pct": dlt, "groups_improved": g,
                           "passes": bool(dlt <= GAIN and g >= MIN_GROUPS)})
            ok = ok and checks[-1]["passes"]
        detail.append({"candidate": m, "current": adopted, "passes": ok, "checks": checks})
        if ok:
            adopted = m
        elif m == "E1":
            break
    if adopted is None:
        adopted = "E1"
    student = "ridge"
    base = CANDIDATES[adopted]
    if base.kind == "student":
        best = scored[adopted]["rmse"]
        for s in LEARNERS[1:]:
            exp = En.Experiment(adopted, "student", groups=list(base.groups), student=s)
            rmse, theta, fold_rmse = middle_score(exp, tr)
            dlt = 100 * (rmse / best - 1)
            g = int(np.sum(np.asarray(fold_rmse) < np.asarray(scored[adopted]["fold_rmse"])))
            if dlt <= GAIN and g >= MIN_GROUPS:
                student, best = s, rmse
                scored[f"{adopted}__{s}"] = {"rmse": rmse, "theta": theta, "fold_rmse": fold_rmse}
    return adopted, student, {"steps": detail}


def _score_fold(args):
    fold, tr, te = args
    dk = En.data_key()
    scored = {}
    for name, exp in CANDIDATES.items():
        rmse, theta, fold_rmse = middle_score(exp, tr, dk)
        scored[name] = {"rmse": rmse, "theta": theta, "fold_rmse": fold_rmse}
        log.info(f"  fold {fold} {name}: middle RMSE={rmse * C.OM_REPORT_FACTOR:.2f}")
    adopted, student, steps = adopt_in_train(tr, scored)
    base = CANDIDATES[adopted]
    key = adopted if student == "ridge" else f"{adopted}__{student}"
    if key not in scored:
        exp = En.Experiment(adopted, base.kind, groups=list(base.groups), student=student,
                            encoder=base.encoder, shuffle=base.shuffle)
        rmse, theta, fold_rmse = middle_score(exp, tr, dk)
        scored[key] = {"rmse": rmse, "theta": theta, "fold_rmse": fold_rmse}
    theta = scored[key]["theta"]
    exp = En.Experiment(adopted, base.kind, groups=list(base.groups), student=student,
                        encoder=base.encoder, shuffle=base.shuffle)
    pred = En.fit_predict(exp, theta, tr, te, dk)
    return {"fold": fold, "adopted": adopted, "student": student, "theta": theta,
            "scores": {k: v["rmse"] for k, v in scored.items()}, "steps": steps,
            "te": te.tolist(), "pred": pred.tolist(),
            "n_train": int(len(tr)), "n_test": int(len(te))}


def main():
    prev = C.RES / "nested_selection.json"
    bak = C.RES / "nested_selection_covariates.json"
    if prev.exists() and not bak.exists():
        bak.write_bytes(prev.read_bytes())
    D = En.get_data()
    outer = outer_splits(D.groups, D.xy, C.BUFFER_MAIN_KM)
    jobs = [(i, tr, te) for i, (tr, te) in enumerate(outer)]
    rows = []
    # Process folds sequentially for memory; within-fold work is already heavy.
    for job in jobs:
        log.info(f"outer fold {job[0]}: scoring {len(CANDIDATES)} candidates "
                 f"(n_train={len(job[1])})")
        rows.append(_score_fold(job))
        log.info(f"  selected {rows[-1]['adopted']} / {rows[-1]['student']}")

    pred = np.full(len(D.y), np.nan)
    for r in rows:
        pred[np.asarray(r["te"], int)] = np.asarray(r["pred"], float)
    mask = ~np.isnan(pred)
    met = En.metrics(D.y[mask], pred[mask])
    def clean(r):
        return {k: v for k, v in r.items() if k not in ("pred", "te")}
    out = {"rule": {"gain_pct": GAIN, "min_groups": MIN_GROUPS, "order": NEST_ORDER,
                    "candidates": list(CANDIDATES), "learners": list(LEARNERS)},
           "folds": [clean(r) for r in rows],
           "metrics": {k: float(v) for k, v in met.items()},
           "group_metrics": {str(int(g)): {k: float(v) for k, v in
                            En.metrics(D.y[D.groups == g], pred[D.groups == g]).items()}
                             for g in np.unique(D.groups)},
           "n": int(mask.sum())}
    save_json(out, C.RES / "nested_selection.json")
    with open(C.OOF / "nested_selection.pkl", "wb") as f:
        pickle.dump({"pred": pred, "folds": rows, "metrics": met}, f)
    # also write a long per-sample table for audit
    pd.DataFrame({"sample_id": D.ids, "outer_group": D.groups,
                  "observed_SOM": D.y * C.OM_REPORT_FACTOR,
                  "prediction": pred * C.OM_REPORT_FACTOR}).to_csv(
        C.RES / "nested_selection_oof.csv", index=False)
    log.info(f"nested selection: R2={met['r2']:.3f} RMSE={met['rmse'] * C.OM_REPORT_FACTOR:.2f} "
             f"choices={[ (r['adopted'], r['student']) for r in rows ]}")


if __name__ == "__main__":
    main()
