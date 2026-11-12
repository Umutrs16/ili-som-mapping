"""Experiment driver: E0-E8 (all students), learning curves, sensitivity analyses,
shuffle null, latent predictability and pH/EC. Each run is cached as a pickle in
outputs/oof/ so the script can be resumed.

    python s05_experiments.py [main|e8|curves|sens|null|latent|aux|all]
"""
from __future__ import annotations

import os
import pickle
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import engine as En
import models as M
from s04_folds import outer_splits
from utils import C, get_logger, save_json

log = get_logger("s05_exp")
B = M.BASE_SETS
STUDENTS = ["ridge", "pls", "krr", "cat", "rf", "tabpfn"]
PUBLIC = "ICRAF-ISRIC"


def E(name, kind, groups=(), **kw):
    return En.Experiment(name=name, kind=kind, groups=list(groups), **kw)


def main_experiments(student: str) -> list[En.Experiment]:
    s = student
    return [
        E("E1", "student", B["E1"], student=s),
        E("E2", "student", B["E2"], student=s),
        E("E3", "student", B["E3"], student=s),
        E("E4a", "transfer", B["E2"], student=s, encoder="local"),
        E("E4b", "transfer", B["E3"], student=s, encoder="local"),
        E("E5a", "transfer", B["E2"], student=s, encoder=PUBLIC),
        E("E5b", "transfer", B["E3"], student=s, encoder=PUBLIC),
        E("E6a-L", "transfer", B["E2"], student=s, encoder="local", shuffle=True),
        E("E6b-L", "transfer", B["E3"], student=s, encoder="local", shuffle=True),
        E("E6a-P", "transfer", B["E2"], student=s, encoder=PUBLIC, shuffle=True),
        E("E6b-P", "transfer", B["E3"], student=s, encoder=PUBLIC, shuffle=True),
        E("E7c-a", "transfer", B["E2"], student=s, encoder=PUBLIC, true_latent=True),
        E("E7c-b", "transfer", B["E3"], student=s, encoder=PUBLIC, true_latent=True),
    ]


def baseline_experiments() -> list[En.Experiment]:
    return [E("E0a", "mean"), E("E0b", "trend"), E("E0c", "kriging"),
            E("Eo", "student", ["O"], student="ridge"),
            E("E7a", "lab_full", student="pls"),
            E("E7a", "lab_full", student="ridge"),
            E("E7b", "lab_sim", student="ridge"), E("E7b", "lab_sim", student="pls"),
            E("E7b", "lab_sim", student="cat")]


def tag_of(exp: En.Experiment, suffix: str = "") -> str:
    # Keep the legacy tag "E7a" for the original PLSR full-spectrum run.
    if exp.kind == "lab_full" and exp.student == "pls":
        return f"{exp.name}{suffix}"
    stud = "" if exp.kind in ("mean", "trend", "kriging") else f"__{exp.student}"
    return f"{exp.name}{stud}{suffix}"


def run_cached(tag: str, exp: En.Experiment, n_jobs: int = 14, **kw) -> dict:
    path = C.OOF / f"{tag}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    res = En.run(exp, n_jobs=n_jobs, **kw)
    res.update(tag=tag, exp=exp.__dict__.copy(), run_kw={k: v for k, v in kw.items()
                                                          if k != "train_subsets"})
    with open(path, "wb") as f:
        pickle.dump(res, f)
    m = res["metrics"]
    log.info(f"{tag:42s} R2={m['r2']:.3f} RMSE={m['rmse']:.3f} CCC={m['ccc']:.3f} "
             f"bias={m['bias']:+.3f} ({res['seconds']:.0f}s)")
    return res


def run_many(items: list[tuple[str, En.Experiment, dict]], n_jobs: int = 14):
    """Parallel over runs (each run sequential inside the worker)."""
    todo = [(t, e, kw) for t, e, kw in items if not (C.OOF / f"{t}.pkl").exists()]
    if todo:
        Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(run_cached)(t, e, 1, **kw) for t, e, kw in todo)
    out = {}
    for t, _, _ in items:
        with open(C.OOF / f"{t}.pkl", "rb") as f:
            out[t] = pickle.load(f)
    log.info(f"run_many: {len(todo)} new / {len(items)} total")
    return out


# ------------------------------------------------------------------ blocks
def stage_main():
    for exp in baseline_experiments():
        run_cached(tag_of(exp), exp)
    for s in [x for x in STUDENTS if x != "tabpfn"] + ["tabpfn"]:
        for exp in main_experiments(s):
            run_cached(tag_of(exp), exp)


def stage_e8():
    exps = [E("E0a", "mean"), E("E7a", "lab_full")]
    for s in ("ridge", "cat"):
        for name in ("E1", "E2", "E3"):
            exps.append(E(name, "student", B[name], student=s))
        for name, base, enc in (("E4a", "E2", "local"), ("E5a", "E2", PUBLIC),
                                ("E5b", "E3", PUBLIC), ("E6a-P", "E2", PUBLIC)):
            exps.append(E(name, "transfer", B[base], student=s, encoder=enc,
                          shuffle=name.startswith("E6")))
    for exp in exps:
        run_cached(tag_of(exp, "__E8random"), exp, scheme="random")


def lc_subsets(size: int, rep: int, buffer_km: float = C.BUFFER_MAIN_KM) -> dict:
    """Block-wise random training subsets of a given size for each outer fold."""
    D = En.get_data()
    rng = np.random.default_rng(C.SEED + 1000 * size + rep)
    subs = {}
    for i, (tr, _) in enumerate(outer_splits(D.groups, D.xy, buffer_km)):
        if size >= len(tr):
            subs[i] = tr
            continue
        blks = np.unique(D.blocks[tr])
        rng.shuffle(blks)
        pick = []
        for b in blks:
            pick.extend(tr[D.blocks[tr] == b].tolist())
            if len(pick) >= size:
                break
        pick = np.array(pick)
        if len(pick) > size:
            last = tr[D.blocks[tr] == b]
            drop = rng.choice(last, size=len(pick) - size, replace=False)
            pick = pick[~np.isin(pick, drop)]
        subs[i] = np.sort(pick)
    return subs


def stage_curves():
    exps = [E("E1", "student", B["E1"]), E("E2", "student", B["E2"]),
            E("E3", "student", B["E3"]),
            E("E4a", "transfer", B["E2"], encoder="local"),
            E("E5a", "transfer", B["E2"], encoder=PUBLIC),
            E("E6a-P", "transfer", B["E2"], encoder=PUBLIC, shuffle=True)]
    items = []
    for size in C.LEARNING_SIZES:
        for rep in range(C.LEARNING_REPEATS):
            subs = lc_subsets(size, rep)
            for exp in exps:
                items.append((tag_of(exp, f"__LC{size}_r{rep}"), exp,
                              {"train_subsets": subs, "seed": C.SEED + rep}))
    run_many(items)


def stage_sens():
    items = []

    def add(tag, exp, **kw):
        items.append((tag, exp, kw))

    core = {"E1": E("E1", "student", B["E1"]), "E2": E("E2", "student", B["E2"]),
            "E3": E("E3", "student", B["E3"]),
            "E4a": E("E4a", "transfer", B["E2"], encoder="local"),
            "E5a": E("E5a", "transfer", B["E2"], encoder=PUBLIC),
            "E5b": E("E5b", "transfer", B["E3"], encoder=PUBLIC),
            "E6a-P": E("E6a-P", "transfer", B["E2"], encoder=PUBLIC, shuffle=True)}
    for bk in (0.0, 10.0, 20.0):
        for k, e in core.items():
            add(tag_of(e, f"__buf{int(bk)}"), e, buffer_km=bk)
        add(f"E0b__buf{int(bk)}", E("E0b", "trend"), buffer_km=bk)
    for var in ("s10_buffer", "s30_buffer", "main_s20_center"):
        for k in ("E1", "E2", "E3", "E5a", "E6a-P"):
            add(tag_of(core[k], f"__{var}"), core[k], dk=En.data_key(variant=var))
    for var in ("ndvi020", "ndvi030", "nbr2", "medoid", "p90", "c10"):
        for k in ("E1", "E2", "E5a", "E6a-P"):
            add(tag_of(core[k], f"__opt-{var}"), core[k], dk=En.data_key(opt_variant=var))
    for k in ("E3", "E5b"):
        add(tag_of(core[k], "__ae2025"), core[k], dk=En.data_key(ae_variant="ae2025"))
    for k in (8, 16):
        for s in ("ridge", "cat"):
            e = E("E3", "student", B["E3"], student=s, ae_pca=k)
            add(tag_of(e, f"__aepca{k}"), e)
    add("E2+XY__ridge", E("E2+XY", "student", B["E2"] + ["XY"]))
    add("E3+XY__ridge", E("E3+XY", "student", B["E3"] + ["XY"]))
    add("E5a+XY__ridge", E("E5a+XY", "transfer", B["E2"] + ["XY"], encoder=PUBLIC))
    add("E2+SG__ridge", E("E2+SG", "student", B["E2"] + ["SG"]))
    add("E2-C__ridge", E("E2-C", "student", ["O", "T", "R", "V"]))
    add("E5a-C__ridge", E("E5a-C", "transfer", ["O", "T", "R", "V"], encoder=PUBLIC))
    for k in ("E1", "E2", "E3", "E5a"):
        for lg in (True, False):
            e = En.Experiment(**{**core[k].__dict__, "log_target": lg})
            add(tag_of(e, f"__log{int(lg)}"), e)
    for d in C.LATENT_DIMS:
        for k in ("E4a", "E5a", "E6a-P"):
            e = En.Experiment(**{**core[k].__dict__, "dims": (d,)})
            add(tag_of(e, f"__d{d}"), e)
    for src in ("LUCAS", "Pooled"):
        for shuf in (False, True):
            nm = "E6a-P" if shuf else "E5a"
            e = E(nm, "transfer", B["E2"], encoder=src, shuffle=shuf)
            add(tag_of(e, f"__src-{src}"), e)
    for prep in ("raw", "sg1d"):
        for k in ("E4a", "E5a", "E6a-P"):
            e = En.Experiment(**{**core[k].__dict__, "prep": prep})
            add(tag_of(e, f"__prep-{prep}"), e)
    for k in ("E4a", "E5a", "E6a-P"):
        e = En.Experiment(**{**core[k].__dict__, "drop_water": True})
        add(tag_of(e, "__nowater"), e)
    run_many(items)


def stage_c10():
    """Post hoc: the deployment rule with the optical composite computed at 10 m and averaged."""
    from s05b_compare import decide
    dk = En.data_key(opt_variant="c10")
    for exp in main_experiments("ridge"):
        if not exp.name.startswith("E7"):
            run_cached(tag_of(exp, "__opt-c10"), exp, dk=dk)
    adopted = decide("__opt-c10")["deployed_experiment"]
    if adopted:
        for s in [x for x in STUDENTS[1:] if x != "tabpfn"] + ["tabpfn"]:
            exp = next(e for e in main_experiments(s) if e.name == adopted)
            run_cached(tag_of(exp, "__opt-c10"), exp, dk=dk)


def stage_null(n_perm: int = 30, students=("ridge",)):
    items = []
    for s in students:
        for base in ("E2", "E3"):
            nm = "E6a-P" if base == "E2" else "E6b-P"
            e = E(nm, "transfer", B[base], student=s, encoder=PUBLIC, shuffle=True)
            for k in range(1, n_perm + 1):
                items.append((tag_of(e, f"__perm{k:02d}"), e, {"seed": C.SEED + 7919 * k}))
    run_many(items)


def stage_latent():
    """Outer-fold predictability of each public latent dimension from satellite covariates."""
    from encoders import load_public_encoders
    from sklearn.metrics import r2_score
    D = En.get_data()
    enc = load_public_encoders()
    rows = []
    for base in ("E1", "E2", "E3"):
        X = D.X[M.features_for(B[base])].to_numpy(float)
        for src in ("ICRAF-ISRIC", "LUCAS", "Pooled", "local"):
            S = D.spectra[("snv", False)]
            Zhat = np.full((len(D.y), 8), np.nan)
            Ztrue = np.full((len(D.y), 8), np.nan)
            for tr, te in outer_splits(D.groups, D.xy, C.BUFFER_MAIN_KM):
                if src == "local":
                    e = M.LocalPCA(8).fit(S[tr])
                else:
                    e = M.FixedPCA(enc[(src, "snv", False)], 8)
                Z = e.transform(S)
                Zhat[te] = M.make_G().fit(X[tr], Z[tr]).predict(X[te])
                Ztrue[te] = Z[te]
            for j in range(8):
                rows.append({"base": base, "encoder": src, "dim": j + 1,
                             "r2_outer": r2_score(Ztrue[:, j], Zhat[:, j]),
                             "r": np.corrcoef(Ztrue[:, j], Zhat[:, j])[0, 1],
                             "corr_with_logOM": np.corrcoef(Ztrue[:, j], np.log(D.y))[0, 1]})
    df = pd.DataFrame(rows)
    df.to_csv(C.RES / "latent_predictability.csv", index=False)
    log.info("latent predictability:\n" + df[df.base == "E2"].pivot_table(
        index="dim", columns="encoder", values="r2_outer").round(3).to_string())


def stage_aux():
    for target in C.AUX_TARGETS:
        dk = En.data_key(target=target)
        exps = [E("E0a", "mean"), E("E0b", "trend"), E("E7a", "lab_full")]
        for s in ("ridge", "cat"):
            exps += [E("E1", "student", B["E1"], student=s),
                     E("E2", "student", B["E2"], student=s),
                     E("E3", "student", B["E3"], student=s),
                     E("E5a", "transfer", B["E2"], student=s, encoder=PUBLIC),
                     E("E6a-P", "transfer", B["E2"], student=s, encoder=PUBLIC, shuffle=True)]
        for exp in exps:
            exp.log_target = False if target == "pH" else exp.log_target
            run_cached(tag_of(exp, f"__{target}"), exp, dk=dk)


def summarise():
    rows = []
    for p in sorted(C.OOF.glob("*.pkl")):
        with open(p, "rb") as f:
            r = pickle.load(f)
        if "tag" not in r or "exp" not in r or "metrics" not in r:
            continue
        m = r["metrics"]
        row = {"tag": r["tag"], "name": r["exp"]["name"], "kind": r["exp"]["kind"],
               "student": r["exp"]["student"], "scheme": r["scheme"],
               "buffer_km": r["buffer_km"], **m,
               "params": str(r["params"]), "n_train": str(r["n_train"]),
               "seconds": round(r["seconds"], 1)}
        for g, gm in r["group_metrics"].items():
            row[f"rmse_g{g}"] = gm["rmse"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(C.RES / "runs_summary.csv", index=False)
    log.info(f"summary: {len(df)} runs -> runs_summary.csv")
    return df


if __name__ == "__main__":
    stages = sys.argv[1:] or ["all"]
    if "all" in stages:
        stages = ["main", "latent", "e8", "curves", "sens", "c10", "null", "aux"]
    fn = {"main": stage_main, "e8": stage_e8, "curves": stage_curves, "sens": stage_sens,
          "c10": stage_c10, "null": stage_null, "latent": stage_latent, "aux": stage_aux}
    for st in stages:
        log.info(f"=== stage {st} ===")
        fn[st]()
    summarise()
