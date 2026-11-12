"""Spatial split conformal prediction, dissimilarity index / area of applicability,
local training-point density, and the H3/H4 analyses.

Conformal rotation: for each test group t, the calibration group is (t+1) mod 5
and the remaining groups (buffered) are used for tuning and fitting. Scores are
absolute residuals (main) or absolute log-ratios (multiplicative variant); q is
the ceil((1-alpha)(m+1))-th smallest score.

DI/AOA follow Meyer & Pebesma (2021): standardised covariates weighted by
permutation importance, DI = nearest-training distance / mean training
distance, threshold = upper whisker of cross-validated training DI. LPD counts
training points within the threshold distance (Schumacher et al., 2025).
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist

import engine as En
import models as M
from s04_folds import buffered_train, conformal_roles, middle_splits_kmeans, outer_splits
from s05_experiments import PUBLIC, E, tag_of
from utils import C, get_logger, load_json, save_json

log = get_logger("s06_unc")
ALPHA = C.CONFORMAL_ALPHA
B = M.BASE_SETS


# ------------------------------------------------------------------ helpers
def conformal_q(scores, alpha=ALPHA):
    m = len(scores)
    k = int(np.ceil((1 - alpha) * (m + 1)))
    return np.inf if k > m else float(np.sort(scores)[k - 1])


def interval_score(y, lo, hi, alpha=ALPHA):
    return (hi - lo) + 2 / alpha * (lo - y) * (y < lo) + 2 / alpha * (y - hi) * (y > hi)


def experiment(name: str, student: str = "ridge") -> En.Experiment:
    spec = {"E1": ("student", "E1", None), "E2": ("student", "E2", None),
            "E3": ("student", "E3", None), "E4a": ("transfer", "E2", "local"),
            "E4b": ("transfer", "E3", "local"), "E5a": ("transfer", "E2", PUBLIC),
            "E5b": ("transfer", "E3", PUBLIC)}[name]
    return E(name, spec[0], B[spec[1]], student=student, encoder=spec[2])


def perm_importance(fitted, X, y, n_rep=10, seed=C.SEED):
    rng = np.random.default_rng(seed)
    base = En.rmse(y, fitted.predict(X))
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        inc = []
        for _ in range(n_rep):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            inc.append(En.rmse(y, fitted.predict(Xp)) - base)
        imp[j] = np.mean(inc)
    return np.clip(imp, 0, None)


class AOA:
    """Weighted-standardised feature space, CV-based DI threshold, LPD."""

    def fit(self, X_tr, weights, cv_labels):
        self.mu = np.nanmedian(X_tr, axis=0)
        Xf = np.where(np.isnan(X_tr), self.mu, X_tr)
        self.m, self.s = Xf.mean(0), Xf.std(0)
        self.s[self.s == 0] = 1.0
        self.w = weights / weights.max() if weights.max() > 0 else np.ones_like(weights)
        self.T = self._tf(Xf)
        self.dbar = float(np.mean(pdist(self.T)))
        d = cdist(self.T, self.T)
        di_cv = np.empty(len(self.T))
        for i in range(len(self.T)):
            other = cv_labels != cv_labels[i]
            di_cv[i] = d[i, other].min() / self.dbar
        self.di_train = di_cv
        q1, q3 = np.percentile(di_cv, [25, 75])
        self.thr = float(di_cv[di_cv <= q3 + 1.5 * (q3 - q1)].max())
        return self

    def _tf(self, X):
        X = np.where(np.isnan(X), self.mu, X)
        return (X - self.m) / self.s * self.w

    def evaluate(self, X):
        d = cdist(self._tf(np.asarray(X, float)), self.T) / self.dbar
        di = d.min(axis=1)
        return {"DI": di, "AOA": di <= self.thr, "LPD": (d <= self.thr).sum(axis=1)}


def fit_aoa(fitted, X_tr, y_tr, xy_tr):
    w = perm_importance(fitted, X_tr, y_tr)
    lab = np.full(len(X_tr), -1)
    idx = np.arange(len(X_tr))
    for k, (_, _, va) in enumerate(middle_splits_kmeans(idx, xy_tr, 0.0)):
        lab[va] = k
    lone = np.where(lab < 0)[0]
    lab[lone] = lab.max() + 1 + np.arange(len(lone))
    return AOA().fit(X_tr, w, lab), w


# ------------------------------------------------------------------ conformal rotation
def conformal_rotation(exp: En.Experiment, dk=En.data_key(), all_calibrations=False):
    D = En.get_data(*dk)
    roles = conformal_roles(D.groups, D.xy, C.BUFFER_MAIN_KM, all_calibrations=all_calibrations)
    thetas = En.select_params(exp, [r["fit"] for r in roles], dk,
                              n_jobs=1 if exp.student == "tabpfn" else 14)
    X = D.X[M.features_for(exp.groups)].to_numpy(float)
    pts, summ = [], []
    for r, th in zip(roles, thetas):
        fitted = En.fit_model(exp, th, r["fit"], dk)
        pc, pt = fitted.predict(X[r["cal"]]), fitted.predict(X[r["test"]])
        yc, yt = D.y[r["cal"]], D.y[r["test"]]
        q_abs = conformal_q(np.abs(yc - pc))
        q_log = conformal_q(np.abs(np.log(yc) - np.log(np.clip(pc, 1e-3, None))))
        lo_a, hi_a = np.clip(pt - q_abs, 0, None), pt + q_abs
        lo_l = np.clip(pt, 1e-3, None) * np.exp(-q_log)
        hi_l = np.clip(pt, 1e-3, None) * np.exp(q_log)
        aoa, _ = fit_aoa(fitted, X[r["fit"]], D.y[r["fit"]], D.xy[r["fit"]])
        ev = aoa.evaluate(X[r["test"]])
        geo = cdist(D.xy[r["test"]], D.xy[r["fit"]]).min(axis=1)
        for k, i in enumerate(r["test"]):
            pts.append({"sample_id": D.ids[i], "test_group": r["test_group"],
                        "cal_group": r["cal_group"], "y": yt[k], "pred": pt[k],
                        "lo_abs": lo_a[k], "hi_abs": hi_a[k], "lo_log": lo_l[k], "hi_log": hi_l[k],
                        "cov_abs": lo_a[k] <= yt[k] <= hi_a[k], "cov_log": lo_l[k] <= yt[k] <= hi_l[k],
                        "is_abs": interval_score(yt[k], lo_a[k], hi_a[k]),
                        "is_log": interval_score(yt[k], lo_l[k], hi_l[k]),
                        "DI": ev["DI"][k], "AOA": bool(ev["AOA"][k]), "LPD": int(ev["LPD"][k]),
                        "geo_km": geo[k], "bare_n": D.X["bare_n"].iloc[i]
                        if "bare_n" in D.X else np.nan})
        summ.append({"test_group": r["test_group"], "cal_group": r["cal_group"],
                     "n_fit": len(r["fit"]), "n_cal": len(r["cal"]), "n_test": len(r["test"]),
                     "q_abs": q_abs, "q_log": q_log, "theta": str(th),
                     "rmse_test": En.rmse(yt, pt),
                     "cov_abs_cfg": float(np.mean((lo_a <= yt) & (yt <= hi_a))),
                     "cov_log_cfg": float(np.mean((lo_l <= yt) & (yt <= hi_l))),
                     "w_abs_cfg": float(np.mean(hi_a - lo_a)),
                     "w_log_cfg": float(np.mean(hi_l - lo_l))})
    pts = pd.DataFrame(pts)
    if all_calibrations:
        return pts, pd.DataFrame(summ)
    summ = pd.DataFrame(summ)
    by = pts.groupby("test_group").agg(cov_abs=("cov_abs", "mean"), cov_log=("cov_log", "mean"),
                                       w_abs=("hi_abs", "mean"), is_abs=("is_abs", "mean"),
                                       is_log=("is_log", "mean"))
    by["w_abs"] = pts.assign(w=pts.hi_abs - pts.lo_abs).groupby("test_group").w.mean()
    by["w_log"] = pts.assign(w=pts.hi_log - pts.lo_log).groupby("test_group").w.mean()
    summ = summ.merge(by, left_on="test_group", right_index=True)
    return pts, summ


# ------------------------------------------------------------------ OOF AOA (H4)
def oof_aoa(exp: En.Experiment, run: dict, dk=En.data_key()):
    D = En.get_data(*dk)
    X = D.X[M.features_for(exp.groups)].to_numpy(float)
    rows, weights = [], []
    for i, (tr, te) in enumerate(outer_splits(D.groups, D.xy, C.BUFFER_MAIN_KM)):
        fitted = En.fit_model(exp, run["params"][i], tr, dk)
        aoa, w = fit_aoa(fitted, X[tr], D.y[tr], D.xy[tr])
        weights.append(w)
        ev = aoa.evaluate(X[te])
        geo = cdist(D.xy[te], D.xy[tr]).min(axis=1)
        for k, j in enumerate(te):
            rows.append({"sample_id": D.ids[j], "group": i, "y": D.y[j],
                         "pred": run["pred"][j], "abs_err": abs(run["pred"][j] - D.y[j]),
                         "DI": ev["DI"][k], "AOA": bool(ev["AOA"][k]), "LPD": int(ev["LPD"][k]),
                         "DI_thr": aoa.thr, "geo_km": geo[k]})
    w = pd.DataFrame(weights, columns=M.features_for(exp.groups)).mean()
    return pd.DataFrame(rows), w


def h4_summary(df: pd.DataFrame, D) -> dict:
    from scipy.stats import spearmanr
    out = {"n_outside": int((~df.AOA).sum()), "n_inside": int(df.AOA.sum())}
    for c in ("DI", "LPD", "geo_km"):
        rho, p = spearmanr(df[c], df.abs_err)
        out[f"spearman_{c}"] = {"rho": float(rho), "p": float(p)}
    if out["n_outside"] >= 3:
        y, p = df.y.to_numpy(), df.pred.to_numpy()
        ins = df.AOA.to_numpy()
        out["rmse_inside"] = En.rmse(y[ins], p[ins])
        out["rmse_outside"] = En.rmse(y[~ins], p[~ins])
        blocks = pd.Series(D.blocks, index=D.ids).loc[df.sample_id].to_numpy()
        rng = np.random.default_rng(C.SEED)
        ub = np.unique(blocks)
        diffs = []
        for _ in range(C.N_BOOT):
            pick = np.concatenate([np.where(blocks == b)[0] for b in rng.choice(ub, len(ub))])
            a, b_ = pick[ins[pick]], pick[~ins[pick]]
            if len(a) > 2 and len(b_) > 2:
                diffs.append(En.rmse(y[b_], p[b_]) - En.rmse(y[a], p[a]))
        out["rmse_diff_ci"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    lpd_q = pd.qcut(df.LPD.rank(method="first"), 3, labels=["low", "mid", "high"])
    out["rmse_by_LPD_tertile"] = {k: En.rmse(g.y, g.pred) for k, g in df.groupby(lpd_q)}
    return out


# ------------------------------------------------------------------ H3
def h3_analysis(students=("ridge", "pls", "krr", "cat", "rf", "tabpfn")):
    D = En.get_data()
    bare = D.X["bare_n"].to_numpy()
    terc = pd.qcut(bare, 3, labels=["low", "mid", "high"]).astype(str)
    rows = []
    for s in students:
        r2, r3 = (En.load_run(tag_of(experiment(n, s))) for n in ("E2", "E3"))
        if r2 is None or r3 is None:
            continue
        for st in ("low", "mid", "high"):
            m = terc == st
            bb = En.block_bootstrap_delta(D.y[m], r2["pred"][m], r3["pred"][m], D.blocks[m])
            rows.append({"student": s, "stratum": st, "n": int(m.sum()),
                         "bare_n_range": f"{bare[m].min():.0f}-{bare[m].max():.0f}",
                         "rmse_E2": En.rmse(D.y[m], r2["pred"][m]),
                         "rmse_E3": En.rmse(D.y[m], r3["pred"][m]), **bb})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ deployment split
def deployment_split(frac=0.20, buffer_km=C.BUFFER_MAIN_KM):
    """Pre-locked calibration set: 20-km blocks drawn within each spatial group."""
    D = En.get_data()
    rng = np.random.default_rng(C.SEED)
    cal = []
    for g in np.unique(D.groups):
        idx = np.where(D.groups == g)[0]
        blks = np.unique(D.blocks[idx])
        rng.shuffle(blks)
        chosen = []
        for b in blks:
            if len(chosen) >= frac * len(idx):
                break
            chosen.extend(idx[D.blocks[idx] == b].tolist())
        cal.extend(chosen)
    cal = np.array(sorted(cal))
    fit = buffered_train(D.xy, np.setdiff1d(np.arange(len(D.y)), cal), cal, buffer_km)
    pd.DataFrame({"sample_id": D.ids, "role": ["cal" if i in set(cal) else
                                                ("fit" if i in set(fit) else "buffer")
                                                for i in range(len(D.y))]}).to_csv(
        C.FOLDS / "deployment_split.csv", index=False)
    return fit, cal


def main():
    dec = load_json(C.RES / "deployment_decision.json")
    dep_name, dep_student = dec["deployed_experiment"], dec["deployed_student"]
    D = En.get_data()
    names = list(dict.fromkeys(["E1", "E2", "E3", "E5a", "E4a", dep_name]))
    allpts, allsum = [], []
    for nm in names:
        for s in dict.fromkeys(["ridge", dep_student]):
            if s != "ridge" and nm != dep_name:
                continue
            exp = experiment(nm, s)
            pts, summ = conformal_rotation(exp)
            pts.insert(0, "model", f"{nm}__{s}")
            summ.insert(0, "model", f"{nm}__{s}")
            allpts.append(pts)
            allsum.append(summ)
            log.info(f"conformal {nm}__{s}: cov_abs={pts.cov_abs.mean():.3f} "
                     f"cov_log={pts.cov_log.mean():.3f} width_abs={(pts.hi_abs-pts.lo_abs).mean():.2f} "
                     f"width_log={(pts.hi_log-pts.lo_log).mean():.2f} "
                     f"IS_abs={pts.is_abs.mean():.2f} IS_log={pts.is_log.mean():.2f}")
    pd.concat(allpts).to_csv(C.RES / "conformal_points.csv", index=False)
    pd.concat(allsum).to_csv(C.RES / "conformal_summary.csv", index=False)

    h4 = {}
    for nm in dict.fromkeys(["E1", "E2", "E3", dep_name]):
        exp = experiment(nm, dep_student if nm == dep_name else "ridge")
        run = En.load_run(tag_of(exp))
        df, w = oof_aoa(exp, run)
        df.to_csv(C.RES / f"oof_aoa_{nm}.csv", index=False)
        w.sort_values(ascending=False).to_csv(C.RES / f"aoa_weights_{nm}.csv")
        h4[nm] = h4_summary(df, D)
        log.info(f"H4 {nm}: {h4[nm]}")
    save_json(h4, C.RES / "h4_summary.json")

    h3 = h3_analysis()
    h3.to_csv(C.RES / "h3_bare_strata.csv", index=False)
    log.info("H3:\n" + h3[["student", "stratum", "n", "bare_n_range", "rmse_E2", "rmse_E3",
                           "delta_pct", "pct_ci_low", "pct_ci_high"]].round(3).to_string())

    fit, cal = deployment_split()
    save_json({"n_fit": len(fit), "n_cal": len(cal),
               "n_buffer": len(D.y) - len(fit) - len(cal),
               "cal_by_group": pd.Series(D.groups[cal]).value_counts().sort_index().to_dict()},
              C.RES / "deployment_split.json")
    log.info(f"deployment split: fit={len(fit)} cal={len(cal)}")


if __name__ == "__main__":
    main()
