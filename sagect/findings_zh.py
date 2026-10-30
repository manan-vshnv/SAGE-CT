"""Finding extraction and clinical scores for Chinese CTRG-Brain findings (complements PCRL's NLG script).

term_f1    : micro P/R/F1 over MEPNet's 32 released medical entities (presence in the report).
finding_f1 : micro P/R/F1 over (lesion, region) pairs from abnormal clauses. A clause counts as abnormal when it
             names a lesion term and is not negated (未见/无/正常/...). Laterality is part of the region
             (左侧基底节区 != 右侧基底节区). A fixed template report scores ~0 here, unlike on BLEU/CIDEr.
lesion_f1  : the same with regions dropped (did the report name the right kinds of abnormality?).
"""
import re

from . import paths

ENT_FILE = f"{paths.MEPNET_DIR}/CTRG_dataset/medical entities.txt"
ENTITIES = [l.strip() for l in open(ENT_FILE, encoding="utf-8") if l.strip() and "Entities" not in l]

LESIONS = {
    "高密度": r"高密度|稍高密度",
    "低密度": r"低密度|稍低密度|密度减低",
    "水肿": r"水肿",
    "出血/血肿": r"出血|血肿",
    "骨折": r"骨折|骨质不连续|线样低密度影.*骨|骨.*线样",
    "积液/积气": r"积液|积气|液平",
    "受压/移位": r"受压|移位|偏移|左移|右移",
    "增宽/萎缩": r"增宽|脑回变小|萎缩",
    "肿胀": r"肿胀",
    "增厚": r"增厚",
    "软化灶": r"软化",
    "钙化": r"钙化",
}
REGIONS = ["基底节", "丘脑", "侧脑室", "第三脑室", "第四脑室", "脑室", "额叶", "顶叶", "颞叶", "枕叶", "岛叶",
           "放射冠", "半卵圆中心", "胼胝体", "小脑", "脑干", "桥脑", "中脑", "延髓", "大脑镰", "颅板下", "硬膜下",
           "蛛网膜下腔", "脑沟", "脑池", "外侧裂", "纵裂", "鞍区", "颅骨", "颅底", "皮下", "上颌窦", "筛窦", "蝶窦",
           "额窦", "乳突", "眼眶", "中线"]
ATROPHY_REGIONS = {"脑沟", "脑池", "蛛网膜下腔", "外侧裂", "纵裂", "其他"}
LOCAL_NEG = re.compile(r"无|未见|未|不伴|没有")  # negation right before a lesion word
NEG = re.compile(r"未见|无明显|无异常|未显示|正常|居中|对称|清晰|无移位|无扩大")


def clauses(text):
    return [c for c in re.split(r"[，。；;,.\n]", text) if c.strip()]


def side_of(c):
    l, r = "左" in c, "右" in c
    if ("双" in c or "两" in c) or (l and r):
        return "双侧"
    return "左" if l else ("右" if r else "")


def findings_of(text):
    out = set()
    for c in clauses(text):
        c = c.replace("增完", "增宽")  # frequent typo in CTRG
        les = [k for k, p in LESIONS.items() if any(not LOCAL_NEG.search(c[max(0, m.start() - 4): m.start()])
                                                    for m in re.finditer(p, c))]
        if not les or (NEG.search(c) and not re.search(r"高密度|低密度|出血|血肿|骨折", c)):
            continue
        regs = [r for r in REGIONS if r in c]
        regs = [r for r in regs if not any(r != o and r in o for o in regs)] or ["其他"]  # drop 脑室 if 侧脑室
        s = side_of(c)
        for l in les:
            if l == "增宽/萎缩" and set(regs) <= ATROPHY_REGIONS:  # one global atrophy finding, not one per sulcus
                out.add((l, "全脑"))
                continue
            for r in regs:
                out.add((l, s + r))
    return out


def prf(pred_sets, gt_sets):
    tp = sum(len(p & g) for p, g in zip(pred_sets, gt_sets))
    np_ = sum(len(p) for p in pred_sets)
    ng = sum(len(g) for g in gt_sets)
    P = tp / np_ if np_ else 0.0
    R = tp / ng if ng else 0.0
    return {"P": round(100 * P, 2), "R": round(100 * R, 2), "F1": round(100 * 2 * P * R / (P + R), 2) if P + R else 0.0}


def clinical_scores(preds, gts):
    terms = lambda t: {e for e in ENTITIES if e in t}  # noqa: E731
    fp = [findings_of(p) for p in preds]
    fg = [findings_of(g) for g in gts]
    return {"term": prf([terms(p) for p in preds], [terms(g) for g in gts]),
            "finding": prf(fp, fg),
            "lesion": prf([{l for l, _ in s} for s in fp], [{l for l, _ in s} for s in fg])}
