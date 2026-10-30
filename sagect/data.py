"""Splits, report text and the annotation file of both datasets (ds = "ctrg" or "indinfm").

ctrg    : CTRG-Brain, Chinese findings, the 7:1:2 split released with PCRL (4200 / 601 / 1200 studies).
indinfm : IndiNFM, English findings, official train / test split, 10% of train held out as validation
          (md5 of the study key), only studies whose slices were converted.
"""
import hashlib
import json
import os
import re
import sys

from . import paths

NO_FINDING = {"ctrg": "未见明显异常", "indinfm": "No focal abnormality"}


def pcrl_scorer(ds):
    """PCRL's metric module for the language (loads PCRL's jieba dictionary for Chinese)."""
    sys.path.insert(0, paths.PCRL_DIR)
    if ds == "ctrg":
        from llama_recipes.pycocoevalcap import metrics_for_chs_mrg as m
    else:
        from llama_recipes.pycocoevalcap import metrics_for_eng_mrg as m
    return m


def clean_en(t):
    return re.sub(r"\s+", " ", (t or "").replace("\n", " ")).strip()


def load_split(ds, name):
    """name in train / validation / test -> [{"id", "findings"(, "images")}]."""
    if ds == "ctrg":
        f = f"{paths.PCRL_DIR}/dataset/CTRG_SAM_SEG_dataset/splits/{name}.json"
        return [{"id": e["id"], "findings": e["findings"], "images": e["images"]} for e in json.load(open(f, encoding="utf-8"))]
    src = "test" if name == "test" else "train"
    out = []
    for inst in json.load(open(f"{paths.INDINFM_DIR}/{src}.json", encoding="utf-8")).values():
        for p in inst.values():
            key = f"{p['institute_id']}_{p['patient_id']}"
            f = clean_en(p.get("findings"))
            if not f or not os.path.exists(paths.work("slices_indinfm", key, "done")):
                continue
            if src == "train":
                in_val = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10 == 0  # stable 10% validation
                if in_val != (name == "validation"):
                    continue
            out.append({"id": key, "findings": f})
    return out


def clauses(ds, t):
    if ds == "ctrg":
        return [c.strip() for c in re.split(r"[，。；;]", t) if c.strip()]
    return [s.strip().rstrip(".").strip() for s in re.split(r"(?<=[.;])\s+", clean_en(t)) if len(s.strip()) > 3]


def tok_en(t):
    return " ".join(re.findall(r"[a-z0-9]+|[^\sa-z0-9]", t.lower()))


def tokens(ds, text):
    """CTRG: jieba words with PCRL's medical dictionary (as the scorer), whitespace removed; IndiNFM: tok_en."""
    if ds == "ctrg":
        import jieba
        pcrl_scorer(ds)
        return [w for w in jieba.cut(re.sub(r"\s+", "", text), cut_all=False, HMM=True) if w.strip()]
    return tok_en(text).split()


def ann_path(ds):
    return paths.work(f"ann_{ds}.json")


def load_ann(ds):
    return json.load(open(ann_path(ds), encoding="utf-8"))


def detok(toks, ds):
    """Model tokens -> plain text in the dataset's style (the scorer re-tokenises)."""
    toks = [t for t in toks if t and t not in ("<unk>", "UNK", "<pad>", "<bos>", "<eos>")]
    if ds == "ctrg":
        return "".join(toks)
    s = " ".join(toks)
    s = re.sub(r"\s+([.,;:!?%)\]])", r"\1", s)
    s = re.sub(r"([(\[])\s+", r"\1", s)
    s = re.sub(r"\s*([/\-])\s*", r"\1", s)
    s = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", s)
    s = re.sub(r"(^|[.;]\s)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)  # sentence case
    return s.strip()
