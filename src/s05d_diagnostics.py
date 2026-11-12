"""Cheap diagnostics that reuse saved out-of-fold predictions.

Block-size sensitivity of the paired bootstrap (20 and 40 km) and the
low-versus-high bare-soil contrast for H3.

    python s05d_diagnostics.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import engine as En
from s04_folds import blocks
from s05b_compare import load, tag
from utils import C, get_logger, save_json

log = get_logger("s05d")


def main():
    D = En.get_data()
    pairs = [("E3", "E1"), ("E3", "E2"), ("E4a", "E6a-L"), ("E5a", "E6a-P"), ("E5b", "E6b-P")]
    rows = []
    for size in (20, 40):
        bid = blocks(D.xy, size)
        for model, ref in pairs:
            ra, rb = load(tag(ref, "ridge")), load(tag(model, "ridge"))
            bb = En.block_bootstrap_delta(D.y, ra["pred"], rb["pred"], bid)
            rows.append({"block_km": size, "model": model, "reference": ref,
                         "n_blocks": int(len(np.unique(bid))), **bb})
    pd.DataFrame(rows).to_csv(C.RES / "block_sensitivity.csv", index=False)

    bare = D.X["bare_n"].to_numpy()
    terc = pd.qcut(bare, 3, labels=["low", "mid", "high"]).astype(str)
    low = terc == "low"
    high = terc == "high"
    r2, r3 = load("E2__ridge"), load("E3__ridge")
    # paired squared-error improvement of E3 over E2
    d = (D.y - r2["pred"]) ** 2 - (D.y - r3["pred"]) ** 2
    bid = blocks(D.xy, 20)
    rng = np.random.default_rng(C.SEED)
    ub = np.unique(bid)
    diffs = []
    for _ in range(C.N_BOOT):
        pick = np.concatenate([np.where(bid == b)[0] for b in rng.choice(ub, len(ub))])
        if low[pick].sum() < 5 or high[pick].sum() < 5:
            continue
        diffs.append(d[pick][low[pick]].mean() - d[pick][high[pick]].mean())
    out = {"low_minus_high_mse_improve": float(d[low].mean() - d[high].mean()),
           "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
           "n_low": int(low.sum()), "n_high": int(high.sum())}
    save_json(out, C.RES / "h3_interaction.json")
    log.info(f"H3 low−high MSE improvement {out['low_minus_high_mse_improve']:.4f} "
             f"CI {out['ci95']}")
    log.info(f"block sensitivity -> block_sensitivity.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
