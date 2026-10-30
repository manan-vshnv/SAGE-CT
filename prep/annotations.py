"""Annotation file of one dataset: work/ann_<ds>.json = {"train"|"val"|"test": [{"id", "report", "tokens"}]}.

  python -m prep.annotations ctrg        (or indinfm; CPU, seconds)
"""
import json
import os
import sys

from sagect import data as D


def main(ds):
    out = {}
    for k, s in (("train", "train"), ("val", "validation"), ("test", "test")):
        out[k] = [{"id": c["id"], "report": c["findings"], "tokens": D.tokens(ds, c["findings"])} for c in D.load_split(ds, s)]
    os.makedirs(os.path.dirname(D.ann_path(ds)), exist_ok=True)
    json.dump(out, open(D.ann_path(ds), "w", encoding="utf-8"), ensure_ascii=False)
    print(ds, {k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main(sys.argv[1])
