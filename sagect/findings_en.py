"""Finding extraction and clinical scores for English IndiNFM findings (same metric definitions as findings_zh).

term_f1    : micro P/R/F1 over finding classes + regions mentioned (presence).
finding_f1 : micro P/R/F1 over (finding class, sided region) pairs from non-negated finding mentions.
lesion_f1  : finding classes only.

Scope: a finding takes the regions that follow it in its clause up to a scope break (extending, with, causing,
compressing, ...) or the next finding; if none, the regions before it; if none, those of the next located finding
in the clause; else "global". Structured headings ("INTRACRANIAL HEMORRHAGE: Nil") and post-negations
("X is not seen") are handled.
"""
import re

NEG = re.compile(r"\b(no|not|without|absence|negative for|rule out|r/o|free of)\b|unremarkable|\bnormal\b")

HEM = r"ha?emorr?h?ag|ha?ematoma|bleed|\bich\b|\biph\b|blood products"
CLASSES = {
    "hypodense": r"low[- ]?density|hypo[- ]?dens|low attenuation|hypo[- ]?attenuat",
    "hyperdense": r"high[- ]?density|hyper[- ]?dens|hyper[- ]?attenuat",
    "edema": r"o?edema",
    "calcification": r"calcif",
    "hemorrhage": HEM,
    "sdh": r"subdural|\bsdh\b",
    "edh": r"extradural|epidural|\bedh\b",
    "sah": r"sub[- ]?arach?noid (?:ha?emorr|bleed|blood)|subarchanoid|\bsah\b",
    "ivh": r"intraventricular|intra-ventricular|\bivh\b",
    "infarct": r"infarct|lacun|ischa?emi|gliosis|gliotic|encephalomalac|small vessel|microangiopath|leukoaraios",
    "contusion": r"contusion",
    "fracture": r"fracture",
    "mass": r"\bmass\b(?! effect)|space[- ]occupying|tumou?r|neoplas|metasta|meningioma|glioma|arachnoid cyst|epidermoid|colloid cyst",
    "sinus_disease": r"sinusitis|mucosal thickening|mucosal opacif|opacification|ha?emosinus|polyp|retention cyst"
                     r"|air[- ]fluid level",
    "atrophy": r"atroph|volume loss|involution|ex[- ]?vacuo|promin[ae]n\w*(?: of)?(?: the)?(?: \w+){0,4} (?:sulc|cistern|sylvian"
               r"|fissure|ventric|folia|csf spaces)|(?:sulc\w*|cisterns?|fissures?|ventricles?|folia|spaces)(?: \w+){0,3} (?:are"
               r"|is|appear|appears|look)? ?(?:mildly |moderately |slightly |diffusely )?promin[ae]n",
    "mass_effect": r"mass effect|midline shift|shift of (?:the )?midline|midline (?:is )?(?:displace|deviat)|effac"
                   r"|(?:subfalcine|uncal|transtentorial|tonsillar) herniation|herniation of (?:the )?(?:uncus|brain)"
                   r"|compress\w* (?:on |of |over )?(?:the )?(?:\w+ ){0,3}ventric|ventric\w* (?:is |are )?compressed",
    "hydrocephalus": r"hydroceph|ventriculomegaly|(?:dilated|dilatation of|dilation of)(?: the)?(?: \w+){0,3} ventric"
                     r"|ventric\w*(?: system)? (?:is |are )?(?:mildly |moderately |markedly )?(?:dilated|enlarged)"
                     r"|transependymal",
    "pneumocephalus": r"pneumoceph|intracranial (?:free )?(?:air|foci)|air[- ]?foci|foci of air|air pockets|pockets of air",
    "scalp_swelling": r"soft tissue swelling|soft tissue thickening|subgaleal|sub galeal|scalp (?:ha?ematoma|swelling)|cephalha?ematoma",
    "postoperative": r"craniotom|craniectom|burr hole|cranioplast|post[- ]?operative|post[- ]?surgical|operated"
                     r"|shunt|clips? (?:in ?situ|noted)|evd\b",
}
# context exclusions for a class (pattern searched in the clause)
EXCLUDE = {
    "hydrocephalus": r"ex[- ]?vacuo",  # ex-vacuo dilatation = volume loss, counted as atrophy
    "pneumocephalus": r"subcutaneous|soft tissue|subgaleal|sub galeal|sinus|mastoid|orbit|emphysema",
    "sinus_disease": r"lung|pleura|bronch",
    "hemorrhage": r"subgaleal|sub galeal|scalp|cephalh",  # extracranial haematoma -> scalp_swelling
    "scalp_swelling": r"sinus|nasal|antr",
}
SEP = r"\s?-?\s?"
REGIONS = {
    "cerebellum": r"cerebell|posterior fossa|vermis|folia",
    "fourth ventricle": r"fourth ventricle|4th ventricle|ivth ventricle",
    "medulla": r"medulla",
    "cerebellopontine angle": r"cerebellopontine|\bcp angle",
    "orbit": r"orbit|retro-?bulbar|lamina papyracea|optic canal|optic nerve",
    "paranasal sinus": r"maxillary sinus|maxillary antr|maxillary and|maxillary,|\bmaxillary sinuses|ethmoid|sphenoid(?:al)? sinus"
                       r"|sphenoid(?:al)? air|frontal sinus|paranasal|nasal cavity|antrum|osteomeatal|\bomu\b",
    "facial bones": r"nasal bone|zygom|maxilla\b|maxillary bone|mandib|pterygoid|nasal septum|facial bone",
    "falx": r"\bfalx|falcine|interhemispheric|\\btentori",
    "mastoid": r"mastoid|middle ear|tympan",
    "skull base": r"skull base|clivus|wing of (?:the )?sphenoid|sphenoid (?:bone|wing)|petrous",
    "pons": r"(?<!pre)(?<!pre-)(?<!pre )\bpons\b|hemi ?pons|(?<!pre)(?<!pre-)(?<!pre )pontine(?! cistern)",
    "midbrain": r"midbrain|mesencephal(?!ic cistern)|cerebral peduncle",
    "brainstem": r"brain ?stem",
    "sella": r"suprasellar(?! cistern)|parasellar|\bsella|pituitary",
    "temporal": r"tempor(?:al|o)(?!mandibular)",
    "frontal": r"front(?:al|o)(?! sinus|alis)|bifrontal",
    "insula": r"sylvian|lateral fissure|insula",
    "basal ganglia": r"basal gangli|lentiform|putamen|putaminal|globus pallidus|internal capsule|external capsule"
                     r"|extreme capsule|caudate|ganglio" + SEP + r"capsular|capsulo" + SEP + r"ganglionic|striat",
    "thalamus": r"thalam",
    "third ventricle": r"third ventricle|3rd ventricle|iiird ventricle",
    "occipital": r"occipit(?:al|o)",
    "lateral ventricle": r"lateral ventricle|frontal horn|occipital horn|temporal horn|atrium of",
    "periventricular": r"peri" + SEP + r"ventricular|para" + SEP + r"ventricular",
    "corpus callosum": r"corpus callosum|callos|splenium|genu",
    "corona radiata": r"corona radiata",
    "parietal": r"pariet(?:al|o)",
    "centrum semiovale": r"centrum semiovale|semioval|semi-oval",
    "vertex": r"vertex|high convexity|parasagittal|para saggital",
}
INSIDE = {"mass_effect", "hydrocephalus"}  # their pattern names the structure ("compression of right lateral ventricle")
# a finding's location span ends at one of these (or at the next finding mention)
STOP = re.compile(r"\b(?:extending|extends|extension|causing|compress\w*|effac\w*|resulting|with|abutting|adjacent|into"
                  r"|reaching|displac\w*|associated|along with|and also|leading|due to|secondary to|s/o)\b")
POST_NEG = re.compile(r"^[^:]*:\s*(?:nil|none|normal|absent|no\b|not seen)|\b(?:not (?:seen|noted|identified|visuali[sz]ed"
                      r"|present|detected|evident|demonstrated)|(?:is|are) absent|(?<!not be )(?<!cannot be )ruled out|nil)\b")
VERB = re.compile(r"\b(?:noted|seen|is|are|was|were|present|shows?|showing|reveals?|identified|demonstrated|observed"
                  r"|suggest\w*|likely|measuring|appears?|involving|s/o|in)\b")
HEADING = re.compile(r"(?<=[a-z.)\]])\s+(?=[A-Z][A-Z0-9 ,/&()\-]{2,}:)")  # "...: Nil MIDLINE SHIFT: Nil"


def side_of(c):
    l, r = re.search(r"\bleft\b", c), re.search(r"\bright\b", c)
    if re.search(r"\b(both|bilateral|bilaterally|bifrontal|biparietal)\b", c) or (l and r):
        return "bilateral "
    return "left " if l else ("right " if r else "")


def sentences(text):
    text = HEADING.sub(". ", (text or "").replace("\n", " "))
    return [s for s in re.split(r"(?<=[.;])\s+", text.lower()) if s.strip()]


def _mentions(clause, negated_scope):
    """[(class, start, end)] of non-negated finding mentions in a clause."""
    out = []
    for k, p in CLASSES.items():
        for m in re.finditer(p, clause):
            if k in EXCLUDE and re.search(EXCLUDE[k], clause) and not re.match(r"pneumoceph|intracranial", m.group()):
                continue
            if negated_scope or NEG.search(clause[: m.start()]) or POST_NEG.search(clause[m.end():]):
                continue
            out.append((k, m.start(), m.end()))
    return sorted(out, key=lambda x: x[1])


SIDE_W = re.compile(r"\b(?:left|right|both|bilateral|bilaterally|bifrontal|biparietal)\b")


def _regions(span):
    """[(name, start)] of the regions named in span, in text order."""
    hits = [(n, m.start()) for n, p in REGIONS.items() for m in [re.search(p, span)] if m]
    return sorted(hits, key=lambda x: x[1])


def _sided(span, regs, default=""):
    """Side per region: the last side word between the previous region and this one ("left thalamus and right
    insula", "bilateral corona radiata, basal ganglia"), else the previous region's side, else `default`."""
    out, prev_end, cur = [], 0, default
    for n, st in regs:
        w = SIDE_W.findall(span[max(prev_end, st - 30): st])
        if w:
            cur = side_of(w[-1])
        out.append((cur, n))
        prev_end = st + 1
    if not any(sd for sd, _ in out):  # "in the frontal lobe on the left"
        out = [(side_of(span), n) for _, n in out]
    return out


ADJ_GAP = re.compile(r"^(?:[\s-]+(?!(?:in|of|at|along|with|and|or|noted|seen|is|are|was|were|shows?|the|from|to|by|on|over"
                     r"|into|causing|resulting|extending|likely|suggest\w*|measuring|s|region|area)\b)[a-z-]+){0,2}[\s-]*$")
# ^ "periventricular white matter hypo density": only modifiers between a region and the finding word
SINUS = {"paranasal sinus", "mastoid"}
VENTRICLES = {"lateral ventricle", "third ventricle", "fourth ventricle"}


def _locate(clause, ms):
    """[(sided regions)] for each mention: after-span up to a scope break (+ adjacent modifiers before it), else the
    before-span back to the previous mention, else the whole after-span up to the next mention."""
    starts = [s for _, s, _ in ms]
    loc = []
    for k, s, e in ms:
        nxt = min([t for t in starts if t >= e] + [len(clause)])
        full = clause[e:nxt]
        stop = STOP.search(full)
        after = (clause[s:e] if k in INSIDE else "") + (full[: stop.start()] if stop else full)
        before = clause[max([ee for _, ss, ee in ms if ee <= s] + [0]):s]
        if k == "ivh":  # "intraventricular extension into bilateral lateral ventricles": the ventricles named nearby
            cand = [(sp, [r for r in _regions(sp) if r[0] in VENTRICLES]) for sp in (full, before)]
            sp, regs = next(((sp, r) for sp, r in cand if r), ("", []))
            loc.append(_sided(sp, regs))
            continue
        regs = _regions(after)
        if regs:
            adj = [(n, m.start()) for n, p in REGIONS.items() for m in re.finditer(p, before)
                   if ADJ_GAP.match(before[m.end():]) and n not in {r for r, _ in regs}]
            sided = _sided(after, regs)
            sided += _sided(before, sorted(adj, key=lambda x: x[1]), sided[0][0])
        else:
            sided = []
            for cand in (before, full):
                regs = _regions(cand)
                if regs:
                    sided = _sided(cand, regs)
                    break
        loc.append(sided)
    return loc


def findings_of(text):
    out = set()
    for sent in sentences(text):
        prev_neg, prev_cls, prev_side, got = False, [], "", {}
        for clause in re.split(r"[,;]| but | however ", sent):
            # comma-list continuation of a negated clause: "no haemorrhage, mass effect or midline shift" / ", or mass"
            inherit = prev_neg and (not VERB.search(clause) or re.match(r"\s*(?:or|nor|and)\b", clause))
            has_neg = bool(NEG.search(clause))
            prev_neg = inherit or (has_neg and not POST_NEG.search(clause))
            ms = _mentions(clause, inherit)
            head = clause[: ms[0][1]] if ms else clause
            regs = _regions(head)
            # list continuation of the previous clause's findings: "infarcts in right parietal, left thalamus and
            # right insula" / "intraparenchymal hemorrhage, measuring 3 cm in the vermis of cerebellum with edema"
            if prev_cls and regs and not NEG.search(head) and (not ms or not prev_located):
                sided = _sided(head, regs, prev_side)
                for k in prev_cls:
                    got.setdefault(k, set()).update(sd + r for sd, r in _fix(k, sided))
                prev_side = sided[-1][0]
            if not ms:
                if not regs or NEG.search(head):
                    prev_cls = []
                continue
            loc = _locate(clause, ms)
            prev_located = True
            for i, (k, s, e) in enumerate(ms):
                sided = loc[i]
                if not sided:  # "chronic infarct with gliosis noted in left occipital lobe": share the next located one
                    sided = next((r for r in loc[i + 1:] if r), None)
                if not sided:
                    sided = [(side_of(clause[e:]) or side_of(clause[:s]), "global")]
                    prev_located = False
                got.setdefault(k, set()).update(sd + r for sd, r in _fix(k, sided))
                prev_side = sided[-1][0]
            prev_cls = list(dict.fromkeys(k for k, _, _ in ms))
        for k, rs in got.items():  # a class located somewhere in the sentence drops its unlocated mentions
            located = {r for r in rs if not r.endswith("global")}
            keep = rs if k == "atrophy" or not located else located
            out.update((k, r) for r in keep if not any(o != r and o.endswith(" " + r) for o in keep))  # unsided dup
    return out


def _fix(k, sided):
    if k == "atrophy":  # prominent sylvian fissures = global volume loss, not an insular finding
        sided = [(sd, r) for sd, r in sided if r != "insula"] or [("", "global")]
    if k == "sinus_disease":  # "frontal, ethmoid and maxillary sinuses" -> sinus, not frontal lobe
        sided = [(sd, r) for sd, r in sided if r in SINUS] or [(sided[0][0] if sided else "", "paranasal sinus")]
    return sided


def terms(text):
    t = text.lower()
    return {k for k, p in CLASSES.items() if re.search(p, t)} | {n for n, p in REGIONS.items() if re.search(p, t)}


def prf(pred_sets, gt_sets):
    tp = sum(len(p & g) for p, g in zip(pred_sets, gt_sets))
    np_, ng = sum(len(p) for p in pred_sets), sum(len(g) for g in gt_sets)
    P = tp / np_ if np_ else 0.0
    R = tp / ng if ng else 0.0
    return {"P": round(100 * P, 2), "R": round(100 * R, 2), "F1": round(100 * 2 * P * R / (P + R), 2) if P + R else 0.0}


def clinical_scores(preds, gts):
    fp, fg = [findings_of(p) for p in preds], [findings_of(g) for g in gts]
    return {"term": prf([terms(p) for p in preds], [terms(g) for g in gts]),
            "finding": prf(fp, fg),
            "lesion": prf([{f for f, _ in s} for s in fp], [{f for f, _ in s} for s in fg])}


def is_abnormal(sentence):
    return bool(findings_of(sentence))
