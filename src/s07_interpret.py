"""Interpretation: grouped permutation importance on the outer test groups,
exact linear SHAP for the deployed model (grouped), and latent-variable checks.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import engine as En
import models as M
from s04_folds import outer_splits
from s05_experiments import tag_of
from s06_uncertainty import experiment
from utils import C, get_logger, load_json, save_json

log = get_logger("s07_int")
N_PERM = 30


def grouped_permutation(exp: En.Experiment, run: dict, n_perm=N_PERM, seed=C.SEED):
    D = En.get_data()
    feats = M.features_for(exp.groups)
    X = D.X[feats].to_numpy(float)
    cols = {g: [feats.index(f) for f in M.GROUPS[g]] for g in exp.groups}
    splits = outer_splits(D.groups, D.xy, C.BUFFER_MAIN_KM)
    fits = [En.fit_model(exp, run["params"][i], tr) for i, (tr, _) in enumerate(splits)]
    base = np.full(len(D.y), np.nan)
    for (tr, te), f in zip(splits, fits):
        base[te] = f.predict(X[te])
    rmse0 = En.rmse(D.y, base)
    rng = np.random.default_rng(seed)
    out = []
    targets = list(cols) + (["Z"] if exp.kind == "transfer" else [])
    for g in targets:
        incs = []
        for _ in range(n_perm):
            pp = np.full(len(D.y), np.nan)
            for (tr, te), f in zip(splits, fits):
                perm = rng.permutation(len(te))
                if g == "Z":
                    m = f.model
                    Xt = X[te]
                    Zt = m.G_.predict(Xt)[perm]
                    pp[te] = m.H_.predict(np.c_[Xt, Zt])
                else:
                    Xp = X[te].copy()
                    Xp[:, cols[g]] = Xp[perm][:, cols[g]]
                    pp[te] = f.predict(Xp)
            incs.append(En.rmse(D.y, pp) - rmse0)
        incs = np.array(incs)
        out.append({"group": g, "label": M.GROUP_LABEL.get(g, g), "n_features":
                    len(cols.get(g, [])) if g != "Z" else int(run["params"][0].get("d", 0)),
                    "delta_rmse_mean": incs.mean(), "delta_rmse_p025": np.percentile(incs, 2.5),
                    "delta_rmse_p975": np.percentile(incs, 97.5),
                    "delta_pct_mean": 100 * incs.mean() / rmse0})
    return pd.DataFrame(out), rmse0, base


def linear_shap():
    dep = load_json(C.MODELS / "deployment_linear.json")
    if not dep.get("linear"):
        return None
    D = En.get_data()
    feats = dep["features"]
    X = D.X[feats].to_numpy(float)
    fit_idx = np.isin(D.ids, dep["fit_ids"])
    coef = np.array(dep["model"]["coef"])
    phi = (X - X[fit_idx].mean(0)) * coef
    df = pd.DataFrame(phi, columns=feats, index=D.ids)
    df.to_csv(C.RES / "shap_linear_values.csv")
    grp = {}
    for g, fs in M.GROUPS.items():
        fs = [f for f in fs if f in feats]
        if fs:
            grp[g] = df[fs].sum(axis=1)
    grp = pd.DataFrame(grp)
    grp.to_csv(C.RES / "shap_grouped_values.csv")
    summ = pd.DataFrame({"mean_abs_shap": df.abs().mean(), "coef": coef,
                         "feature_sd": X[fit_idx].std(0)}).sort_values("mean_abs_shap",
                                                                        ascending=False)
    summ.to_csv(C.RES / "shap_feature_summary.csv")
    log.info("grouped mean |SHAP| (model scale):\n" + grp.abs().mean().sort_values(
        ascending=False).round(4).to_string())
    return summ


def latent_checks():
    from encoders import load_public_encoders
    D = En.get_data()
    enc = load_public_encoders()
    S = D.spectra[("snv", False)]
    rows = []
    for src in ("ICRAF-ISRIC", "LUCAS", "Pooled"):
        pca = enc[(src, "snv", False)]
        Z = pca.transform(S)
        for j in range(8):
            r = {"encoder": src, "dim": j + 1, "evr_source": pca.explained_variance_ratio_[j],
                 "var_local": Z[:, j].var(), "corr_logOM": np.corrcoef(Z[:, j], np.log(D.y))[0, 1]}
            for c in ("B4", "B11", "BSI", "elev", "MAT", "MAP", "VV_med", "NDVI_mean"):
                r[f"corr_{c}"] = np.corrcoef(Z[:, j], D.X[c])[0, 1]
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(C.RES / "latent_correlations.csv", index=False)
    return df


def main():
    dec = load_json(C.RES / "deployment_decision.json")
    dep_name, dep_student = dec["deployed_experiment"], dec["deployed_student"]
    todo = list(dict.fromkeys([(dep_name, dep_student), ("E3", "ridge"), ("E5b", "ridge"),
                               ("E5a", "ridge"), ("E3", "cat")]))
    tabs = []
    for nm, s in todo:
        exp = experiment(nm, s)
        run = En.load_run(tag_of(exp))
        if run is None:
            continue
        df, r0, base = grouped_permutation(exp, run)
        assert np.allclose(base, run["pred"], rtol=1e-6, atol=1e-6), f"refit mismatch {nm}/{s}"
        df.insert(0, "model", f"{nm}__{s}")
        df["rmse_base"] = r0
        tabs.append(df)
        log.info(f"permutation importance {nm}/{s} (RMSE {r0:.3f}):\n"
                 + df[["group", "delta_rmse_mean", "delta_pct_mean"]].round(3).to_string())
    pd.concat(tabs).to_csv(C.RES / "grouped_permutation_importance.csv", index=False)
    linear_shap()
    latent_checks()


if __name__ == "__main__":
    main()
