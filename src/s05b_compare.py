"""Paired comparisons (block bootstrap), hypothesis checks H1/H2 and the
pre-specified deployment rule.

Adoption rule: a candidate passes when its pooled spatial RMSE is >= 5 % lower
than a reference and it is better in >= 3 of 5 spatial groups. Candidates are
visited in ORDER and adopted sequentially: a candidate must pass against its own
references (transfer models: the same-input baseline and the within-fold shuffle
control E6) and against the currently adopted model. E1 is adopted first if it
passes against the trend baseline. Another learner replaces Ridge only if it
passes the same rule against Ridge on the adopted configuration.

The rule is also applied, post hoc, to the runs with the optical composite
computed at 10 m and averaged to the 20 m grid (suffix __opt-c10, s05 c10).
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import engine as En
from utils import C, get_logger, save_json

log = get_logger("s05b_compare")
STUDENTS = ["ridge", "pls", "krr", "cat", "rf", "tabpfn"]
GAIN = -5.0
MIN_GROUPS = 3

REFS = {"E1": ["E0b"], "E2": ["E1"], "E3": ["E2"],
        "E4a": ["E2", "E6a-L"], "E5a": ["E2", "E6a-P"],
        "E4b": ["E3", "E6b-L"], "E5b": ["E3", "E6b-P"]}
EXTRA = [("E3", "E1"), ("E5a", "E4a"), ("E5b", "E4b"), ("E5a", "E1"), ("E5b", "E1"),
         ("E4a", "E1"), ("E7c-a", "E5a"), ("E7c-b", "E5b"), ("E1", "E0a"), ("E1", "E0c"),
         ("E7b", "E7a")]
ORDER = ["E1", "E2", "E3", "E5a", "E5b", "E4a", "E4b"]
BASELINES = ("E0a", "E0b", "E0c", "E7a")


def load(tag):
    p = C.OOF / f"{tag}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def tag(name, student):
    return name if name in BASELINES else f"{name}__{student}"


def compare(y, blocks, groups, ra, rb, label_a, label_b, student):
    bb = En.block_bootstrap_delta(y, ra["pred"], rb["pred"], blocks)
    ng = En.group_direction(y, ra["pred"], rb["pred"], groups)
    return {"student": student, "model": label_b, "reference": label_a,
            "rmse_ref": ra["metrics"]["rmse"], "rmse_model": rb["metrics"]["rmse"],
            **bb, "groups_improved": ng,
            "passes_rule": bool(bb["delta_pct"] <= GAIN and ng >= MIN_GROUPS)}


def decide(suffix=""):
    """Sequential adoption (primary learner: ridge), then the learner check.

    `suffix` selects a variant of the runs; the baselines do not depend on it."""
    D = En.get_data()
    y, blocks, groups = D.y, D.blocks, D.groups
    missing = []

    def run_of(name, s="ridge"):
        t = tag(name, s) + ("" if name in BASELINES else suffix)
        r = load(t)
        if r is None:
            missing.append(t)
        return r

    steps, adopted = [], None
    for m in ORDER:
        refs = list(REFS[m]) + ([adopted] if adopted and adopted not in REFS[m] else [])
        cand, ref_runs = run_of(m), [run_of(r) for r in refs]
        if cand is None or any(r is None for r in ref_runs):
            break
        checks = [compare(y, blocks, groups, rr, cand, r, m, "ridge") for r, rr in zip(refs, ref_runs)]
        ok = all(c["passes_rule"] for c in checks)
        steps.append({"candidate": m, "current": adopted, "references": refs,
                      "passes": ok, "checks": checks})
        if ok:
            adopted = m
        elif m == "E1":
            break
    deployed = adopted if not missing else None
    student, alt = "ridge", []
    if deployed is not None:
        ridge_run = run_of(deployed)
        for s in STUDENTS[1:]:
            r = run_of(deployed, s)
            if r is not None:
                alt.append(compare(y, blocks, groups, ridge_run, r, f"{deployed}-ridge",
                                   f"{deployed}-{s}", s))
        alt_ok = [c for c in alt if c["passes_rule"]]
        if alt_ok:
            student = min(alt_ok, key=lambda c: c["rmse_model"])["student"]
    final = run_of(deployed, student) if deployed else None
    vs_base = [compare(y, blocks, groups, load(b), final, b, f"{deployed}-{student}", student)
               for b in ("E0a", "E0b", "E0c")] if final else []
    return {"deployed_experiment": deployed if not missing else adopted,
            "deployed_student": student, "deployed_rmse": final["metrics"]["rmse"] if final else None,
            "steps": steps, "student_checks": alt, "deployed_vs_baselines": vs_base,
            "complete": not missing, "missing": missing}


def main():
    D = En.get_data()
    y, blocks, groups = D.y, D.blocks, D.groups
    rows = []
    for s in STUDENTS:
        pairs = [(ref, m) for m, refs in REFS.items() for ref in refs] + \
                [(b, a) for a, b in EXTRA]
        for ref, m in pairs:
            ra, rb = load(tag(ref, s)), load(tag(m, s))
            if m == "E7b" and rb is None:
                rb = load(f"E7b__{s}")
            if ra is None or rb is None:
                continue
            rows.append(compare(y, blocks, groups, ra, rb, ref, m, s))
    cmp = pd.DataFrame(rows)
    cmp.to_csv(C.RES / "comparisons.csv", index=False)

    # hypothesis summaries per learner
    hyp = []
    for s in STUDENTS:
        sub = cmp[cmp.student == s]
        if sub.empty:
            continue

        def ok(m, r):
            q = sub[(sub.model == m) & (sub.reference == r)]
            return bool(q.passes_rule.iloc[0]) if len(q) else None

        hyp.append({"student": s,
                    "H1_E4a": ok("E4a", "E2") and ok("E4a", "E6a-L"),
                    "H1_E5a": ok("E5a", "E2") and ok("E5a", "E6a-P"),
                    "H1_E4b": ok("E4b", "E3") and ok("E4b", "E6b-L"),
                    "H1_E5b": ok("E5b", "E3") and ok("E5b", "E6b-P"),
                    "H2_a": ok("E5a", "E4a"), "H2_b": ok("E5b", "E4b"),
                    "E3_vs_E2": ok("E3", "E2")})
    hyp = pd.DataFrame(hyp)
    hyp.to_csv(C.RES / "hypotheses_H1_H2.csv", index=False)

    decision = decide()
    decision["c10"] = decide("__opt-c10")
    decision["rule"] = {"gain_pct": GAIN, "min_groups": MIN_GROUPS, "order": ORDER,
                        "references": REFS, "sequential": True}
    save_json(decision, C.RES / "deployment_decision.json")
    for d, what in ((decision, "main"), (decision["c10"], "c10")):
        for st in d["steps"]:
            log.info(f"[{what}] step {st['candidate']:4s} vs {st['references']} -> {st['passes']}")
        for c in d["student_checks"] + d["deployed_vs_baselines"]:
            log.info(f"[{what}] {c['model']} vs {c['reference']}: {c['delta_pct']:+.1f}% "
                     f"[{c['pct_ci_low']:+.1f}, {c['pct_ci_high']:+.1f}] groups "
                     f"{c['groups_improved']} -> {c['passes_rule']}")
        log.info(f"[{what}] deployed: {d['deployed_experiment']} with {d['deployed_student']}"
                 f"{'' if d['complete'] else ' (incomplete: ' + ', '.join(d['missing']) + ')'}")
    log.info("\n" + hyp.to_string())
    show = cmp[cmp.student == "ridge"][["model", "reference", "rmse_ref", "rmse_model",
                                         "delta_pct", "pct_ci_low", "pct_ci_high",
                                         "groups_improved", "passes_rule"]]
    log.info("\n" + show.round(3).to_string())


if __name__ == "__main__":
    main()
