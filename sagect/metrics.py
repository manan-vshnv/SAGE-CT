"""Scoring: PCRL's NLG scripts (BLEU, METEOR, ROUGE-L, CIDEr) and finding F1, all x100.

METEOR needs a Java runtime on PATH.

  python -m sagect.metrics --ds ctrg preds.json      (preds.json = [{"id", "pred", "gt"}])
"""
import argparse
import json
import os

from . import data

TARGET = {"BLEU_4": 34.4, "METEOR": 34.6, "ROUGE_L": 50.4, "CIDEr": 86.6}  # normalisers of the selection score
KEEP = ("BLEU_4", "METEOR", "ROUGE_L", "CIDEr", "finding_F1", "normal_rate_pred", "distinct_preds", "select_score")


def extractor(ds):
    if ds == "ctrg":
        from . import findings_zh as ext
    else:
        from . import findings_en as ext
    return ext


def score(preds, gts, ds):
    preds, gts = [p.strip() for p in preds], [g.strip() for g in gts]
    g = {k: [x] for k, x in enumerate(gts)}
    r = {k: [p if p else data.NO_FINDING[ds]] for k, p in enumerate(preds)}
    if ds != "ctrg":
        g = {k: [data.tok_en(v[0])] for k, v in g.items()}
        r = {k: [data.tok_en(v[0])] for k, v in r.items()}
    out = {m: round(100 * v, 2) for m, v in data.pcrl_scorer(ds).compute_scores(g, r).items()}
    ext = extractor(ds)
    out.update({f"{k}_{m}": v for k, d in ext.clinical_scores(preds, gts).items() for m, v in d.items()})
    out["normal_rate_pred"] = round(100 * sum(not ext.findings_of(p) for p in preds) / len(preds), 2)
    out["normal_rate_ref"] = round(100 * sum(not ext.findings_of(x) for x in gts) / len(gts), 2)
    out["distinct_preds"] = len(set(preds))
    out["select_score"] = round(sum(out[k] / v for k, v in TARGET.items()), 4)
    out["n"] = len(preds)
    return out


def score_file(rows, path, ds):
    """rows [{"id", "pred", "gt"}] -> path and path.scores.json; returns the main scores."""
    assert rows, f"no predictions for {path}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(rows, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    s = score([r["pred"] for r in rows], [r["gt"] for r in rows], ds)
    json.dump(s, open(path.rsplit(".", 1)[0] + ".scores.json", "w"), indent=1)
    return {k: s[k] for k in KEEP if k in s}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["ctrg", "indinfm"])
    ap.add_argument("preds")
    a = ap.parse_args()
    rows = json.load(open(a.preds, encoding="utf-8"))
    print(json.dumps(score([r["pred"] for r in rows], [r["gt"] for r in rows], a.ds)))
