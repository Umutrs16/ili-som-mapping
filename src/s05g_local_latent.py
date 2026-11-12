"""Replace pooled local-PCA rows with the unweighted mean of within-fold statistics.

Public encoders stay pooled: their axes are fixed. Local axes are not aligned
across folds, so scores are not concatenated.

    python s05g_local_latent.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import engine as En
import models as M
from s04_folds import outer_splits
from utils import C, get_logger

log = get_logger("s05g")


def main():
    D = En.get_data()
    path = C.RES / "latent_predictability.csv"
    df = pd.read_csv(path)
    S = D.spectra[("snv", False)]
    y = np.log(D.y)
    rows = []
    for base in ("E1", "E2", "E3"):
        X = D.X[M.features_for({"E1": ["O", "T"], "E2": ["O", "T", "R", "V", "C"],
                                "E3": ["O", "T", "R", "V", "C", "A"]}[base])].to_numpy(float)
        fold_r2 = [[] for _ in range(8)]
        fold_r = [[] for _ in range(8)]
        for tr, te in outer_splits(D.groups, D.xy, C.BUFFER_MAIN_KM):
            e = M.LocalPCA(8).fit(S[tr])
            Ztr, Zte = e.transform(S[tr]), e.transform(S[te])
            Zhat = M.make_G().fit(X[tr], Ztr).predict(X[te])
            for j in range(8):
                fold_r2[j].append(r2_score(Zte[:, j], Zhat[:, j]))
                fold_r[j].append(float(np.corrcoef(Zte[:, j], y[te])[0, 1]))
        for j in range(8):
            rows.append({"base": base, "encoder": "local", "dim": j + 1,
                         "r2_outer": float(np.mean(fold_r2[j])),
                         "r": float(np.mean(fold_r[j])),
                         "corr_with_logOM": float(np.mean(fold_r[j])),
                         "summary": "unweighted mean of 5 folds"})
    new = pd.DataFrame(rows)
    keep = df[df.encoder != "local"].copy()
    if "summary" not in keep.columns:
        keep["summary"] = "pooled outer-fold predictions"
    out = pd.concat([keep, new], ignore_index=True)
    out.to_csv(path, index=False)
    log.info("local PC1–2 on E2:\n" + new[(new.base == "E2") & (new.dim <= 2)].to_string(index=False))


if __name__ == "__main__":
    main()
