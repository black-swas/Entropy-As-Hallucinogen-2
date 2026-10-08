#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
combine_models.py -- compare / ensemble the cached features of several experiment_v2.py runs.

No GPU, no model download: it only reads the `feats_*.pkl` caches that experiment_v2.py wrote
(same data order and shuffle seed required -- this is checked).

Answers three questions the single-model reports cannot:
  1. Do the models make the SAME mistakes?  (error overlap, "either is right" ceiling)
  2. Does averaging their answer distributions beat the best single model?  (unweighted: no fitting)
  3. Does cross-model DISAGREEMENT flag wrong answers better than one model's own confidence?

  python combine_models.py --caches out_v2/feats_A.pkl out_v2/feats_B.pkl --names 4B 8B
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_v2 as E  # noqa: E402


def load(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return [d[i] for i in range(len(d))]


def dists(recs):
    return [np.exp(E.ens_letter(r)) for r in recs]          # S4 answer distribution per question


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * (np.log(np.maximum(a, 1e-300)) - np.log(np.maximum(b, 1e-300))), 0.0)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def analyse(recs_list, names, out_dir=None):
    N = len(recs_list[0])
    for nm, rs in zip(names, recs_list):
        assert len(rs) == N, f"{nm}: {len(rs)} questions != {N}"
    for i in range(N):
        g = {rs[i]["gold"] for rs in recs_list}
        n = {rs[i]["n"] for rs in recs_list}
        assert len(g) == 1 and len(n) == 1, (
            f"question {i} differs between caches (gold {g}, n {n}): different seed/limit/data?")
    gold = np.array([r["gold"] for r in recs_list[0]])
    P = [dists(rs) for rs in recs_list]                       # [model][question] -> prob vector
    pred = [np.array([int(np.argmax(p)) for p in Pm]) for Pm in P]
    ok = [(pr == gold).astype(float) for pr in pred]

    L = ["# Cross-model comparison (S4 answer distributions)\n", f"N = {N}\n",
         "| Model | S4 accuracy | 95% CI |", "|---|---|---|"]
    for nm, c in zip(names, ok):
        lo, hi = E.bootstrap_ci(c)
        L.append(f"| {nm} | {c.mean():.4f} | [{lo:.3f}, {hi:.3f}] |")

    # unweighted ensemble of all models
    Pens = [np.mean([P[m][i] for m in range(len(P))], axis=0) for i in range(N)]
    ens_ok = np.array([float(np.argmax(p) == g) for p, g in zip(Pens, gold)])
    lo, hi = E.bootstrap_ci(ens_ok)
    best = int(np.argmax([c.mean() for c in ok]))
    L.append(f"| **ensemble (mean of all)** | {ens_ok.mean():.4f} | [{lo:.3f}, {hi:.3f}] |")
    L.append(f"\nEnsemble minus best single ({names[best]}): {ens_ok.mean() - ok[best].mean():+.4f}, "
             f"McNemar p = {E.mcnemar_p(ens_ok, ok[best]):.3g}\n")

    # error overlap (pairwise)
    L.append("## Do the models make the same mistakes?\n")
    L.append("| Pair | both right | only first right | only second right | both wrong | either right (ceiling) | "
             "same wrong answer given |")
    L.append("|---|---|---|---|---|---|---|")
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            both = ((ok[a] == 1) & (ok[b] == 1)).mean()
            oa = ((ok[a] == 1) & (ok[b] == 0)).mean()
            ob = ((ok[a] == 0) & (ok[b] == 1)).mean()
            bw = ((ok[a] == 0) & (ok[b] == 0)).mean()
            same_wrong = ((ok[a] == 0) & (ok[b] == 0) & (pred[a] == pred[b])).sum() / max(1, ((ok[a] == 0) & (ok[b] == 0)).sum())
            L.append(f"| {names[a]} vs {names[b]} | {both:.3f} | {oa:.3f} | {ob:.3f} | {bw:.3f} | "
                     f"{both + oa + ob:.3f} | {same_wrong:.3f} |")
    L.append("\n'same wrong answer given' = of the questions both get wrong, the share where they pick the SAME "
             "wrong option. High values mean shared misconceptions, which no ensemble can fix.\n")

    # uncertainty signals
    L.append("## Which signal predicts that the ENSEMBLE answer is wrong?  (AUROC; 0.5 = none)\n")
    L.append("| Signal | AUROC | 95% CI | acc@100% | @80% | @60% | @40% | @20% |")
    L.append("|---|---|---|---|---|---|---|---|")
    sigs = {"ensemble max-prob": np.array([p.max() for p in Pens])}
    for m, nm in enumerate(names):
        sigs[f"{nm} own max-prob (vs ensemble correctness)"] = np.array([p.max() for p in P[m]])
    if len(names) >= 2:
        d = np.array([np.mean([js(P[a][i], P[b][i]) for a in range(len(P)) for b in range(a + 1, len(P))])
                      for i in range(N)])
        sigs["-cross-model disagreement (JS divergence)"] = -d
        agree = np.array([float(len({int(np.argmax(P[m][i])) for m in range(len(P))}) == 1) for i in range(N)])
        sigs["all models pick the same option"] = agree + 1e-3 * sigs["ensemble max-prob"]   # tie-break by confidence
    for k, v in sigs.items():
        lo, hi = E.auroc_ci(v, ens_ok)
        a = E.accuracy_at_coverage(v, ens_ok)
        L.append(f"| {k} | {E.auroc(v, ens_ok):.3f} | [{lo:.3f}, {hi:.3f}] | "
                 f"{a[1.0]:.3f} | {a[0.8]:.3f} | {a[0.6]:.3f} | {a[0.4]:.3f} | {a[0.2]:.3f} |")
    text = "\n".join(L)
    print(text)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "combine_models.md"), "w", encoding="utf-8") as f:
            f.write(text)
    return dict(ens_acc=float(ens_ok.mean()), accs=[float(c.mean()) for c in ok])


def selftest():
    recs_a = E.synthetic_records(N=300, seed=3, with_ff=False)
    rng = np.random.default_rng(9)
    recs_b = []
    for r in recs_a:                                           # same questions, independently noisy "second model"
        r2 = dict(r)
        L = r["letter_lp"] + rng.normal(0, 0.9, r["letter_lp"].shape)
        r2["letter_lp"] = L - np.logaddexp.reduce(L, axis=-1, keepdims=True)
        recs_b.append(r2)
    out = analyse([recs_a, recs_b], ["A", "B"])
    assert out["ens_acc"] >= min(out["accs"]) - 0.02
    bad = [dict(r) for r in recs_b]
    bad[5] = dict(bad[5], gold=(bad[5]["gold"] + 1) % bad[5]["n"])
    try:
        analyse([recs_a, bad], ["A", "B"])
        raise SystemExit("alignment check failed to fire")
    except AssertionError:
        pass
    print("\n[selftest] OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caches", nargs="+")
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--out_dir", default="out_v2")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.caches or len(a.caches) < 2:
        sys.exit("give at least two --caches")
    names = a.names or [os.path.basename(p) for p in a.caches]
    assert len(names) == len(a.caches)
    analyse([load(p) for p in a.caches], names, a.out_dir)


if __name__ == "__main__":
    main()
