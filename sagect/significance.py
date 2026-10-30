"""Paired bootstrap test on the test set between two systems, each given by one or more prediction files (seeds).

Per test study, with the same tokenisation and scorers as metrics.py:
  BLEU-4   n-gram guess / correct counts + lengths (corpus BLEU rebuilt exactly from the sums)
  METEOR   per-segment METEOR sufficient statistics; corpus METEOR = METEOR's own EVAL of their sum
  ROUGE-L  per study (corpus = mean)
  CIDEr-D  per study with the document frequencies of the full test references (corpus = mean)
  F1       finding F1: per-study (tp, #pred, #ref), micro F1 rebuilt from the sums
A system with several seeds is one system whose per-study statistics are pooled over seeds.
p (one-sided) = share of B resamples of test studies in which the better system on the full set is not better.

  python -m sagect.significance --ds ctrg --a runs/sagect_ctrg_s*/pred_test.json --b baseline_pred_test.json
"""
import argparse
import json
import math

import numpy as np

from . import data as D
from .metrics import extractor

METRICS = ["BLEU_4", "METEOR", "ROUGE_L", "CIDEr", "finding_F1"]
MET = None  # one METEOR process (java)


def load(path):
    rows = json.load(open(path, encoding="utf-8"))
    if isinstance(rows, dict):
        rows = [{"id": k, **v} for k, v in rows.items()]
    return {str(r["id"]): (r["pred"].strip(), r["gt"].strip()) for r in rows}


def prep(t, ds):
    if ds == "ctrg":
        return D.pcrl_scorer(ds).add_space_to_each_word_Jieba(t).replace("\n", " ").replace("\r", " ")
    return D.tok_en(t)


def per_scan(rows, ids, ds):
    from llama_recipes.pycocoevalcap.bleu.bleu_scorer import cook_refs, cook_test
    from llama_recipes.pycocoevalcap.cider.cider import Cider
    from llama_recipes.pycocoevalcap.rouge import Rouge
    preds = [rows[i][0] or D.NO_FINDING[ds] for i in ids]
    gts = [rows[i][1] for i in ids]
    P, G = [prep(p, ds) for p in preds], [prep(g, ds) for g in gts]
    bleu = np.zeros((len(ids), 10))  # testlen, reflen, guess x4, correct x4
    for k, (p, g) in enumerate(zip(P, G)):
        c = cook_test(p, cook_refs([g], n=4), eff="closest", n=4)
        bleu[k] = [c["testlen"], c["reflen"], *c["guess"], *c["correct"]]
    gd, rd = {k: [g] for k, g in enumerate(G)}, {k: [p] for k, p in enumerate(P)}
    met = np.array([[float(x) for x in MET._stat(rd[k][0], gd[k]).split()] for k in range(len(P))])
    _, rou = Rouge().compute_score(gd, rd)
    _, cid = Cider().compute_score(gd, rd)
    ext = extractor(ds)
    f1 = np.array([[len(a & b), len(a), len(b)] for a, b in
                   ((ext.findings_of(p), ext.findings_of(g)) for p, g in zip(preds, gts))], dtype=float)
    return {"bleu": bleu, "met": np.asarray(met, float), "rou": np.asarray(rou, float), "cid": np.asarray(cid, float),
            "f1": f1}


def merge(stats):
    """several seeds -> one system: counts summed, per-study scores averaged."""
    return {k: (np.sum if k in ("bleu", "f1", "met") else np.mean)([s[k] for s in stats], axis=0) for k in stats[0]}


def meteor_eval(stats):
    """corpus METEOR of summed sufficient statistics, computed by the METEOR jar itself."""
    MET.meteor_p.stdin.write("EVAL ||| " + " ".join(repr(float(x)) for x in stats) + "\n")
    MET.meteor_p.stdout.readline()  # segment score of the single (aggregated) line
    return float(MET.meteor_p.stdout.readline().strip())


def corpus(st, idx):
    b = st["bleu"][idx].sum(0)
    tl, rl, guess, corr = b[0], b[1], b[2:6], b[6:10]
    bl = 1.0
    for k in range(4):
        bl *= (corr[k] + 1e-15) / (guess[k] + 1e-9)
    bl = bl ** 0.25
    if tl < rl:
        bl *= math.exp(1 - (rl + 1e-9) / (tl + 1e-15))
    tp, npd, ngt = st["f1"][idx].sum(0)
    P, Rc = (tp / npd if npd else 0.0), (tp / ngt if ngt else 0.0)
    return {"BLEU_4": 100 * bl, "METEOR": 100 * meteor_eval(st["met"][idx].sum(0)), "ROUGE_L": 100 * st["rou"][idx].mean(),
            "CIDEr": 100 * st["cid"][idx].mean(), "finding_F1": 100 * 2 * P * Rc / (P + Rc) if P + Rc else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["ctrg", "indinfm"])
    ap.add_argument("--a", nargs="+", required=True, help="prediction files of system A (one per seed)")
    ap.add_argument("--b", nargs="+", required=True, help="prediction files of system B")
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    D.pcrl_scorer(a.ds)
    from llama_recipes.pycocoevalcap.meteor import Meteor
    global MET
    MET = Meteor()
    rows = {"A": [load(f) for f in a.a], "B": [load(f) for f in a.b]}
    ids = sorted(set.intersection(*[set(r) for rs in rows.values() for r in rs]))
    st = {n: merge([per_scan(r, ids, a.ds) for r in rs]) for n, rs in rows.items()}
    boot = np.random.default_rng(0).integers(0, len(ids), size=(a.B, len(ids)))
    vals = {n: np.array([[corpus(s, b)[m] for m in METRICS] for b in boot]) for n, s in st.items()}
    full = {n: corpus(s, np.arange(len(ids))) for n, s in st.items()}
    res = {"n_studies": len(ids), "B": a.B}
    for j, m in enumerate(METRICS):
        d = full["A"][m] - full["B"][m]
        diff = vals["A"][:, j] - vals["B"][:, j]
        p = float(np.mean(diff <= 0)) if d > 0 else float(np.mean(diff >= 0))
        res[m] = {"A": round(full["A"][m], 2), "B": round(full["B"][m], 2), "diff": round(d, 2), "p": round(p, 4),
                  "ci95": [round(float(np.percentile(diff, 2.5)), 2), round(float(np.percentile(diff, 97.5)), 2)]}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
