"""Frozen BERT embeddings ([CLS] of the last layer, L2-normalised) of every report clause, used for Stage-1
alignment and as knowledge tokens: bert-base-chinese (CTRG-Brain), BiomedVLP-CXR-BERT-general (IndiNFM).

  python -m prep.clause_emb ctrg     -> work/clause_emb_ctrg.pt : {id: {"emb": (n, 768) float16}}
"""
import sys

import torch
from transformers import AutoModel, AutoTokenizer

from sagect import data as D
from sagect import paths

NAME = {"ctrg": "google-bert/bert-base-chinese", "indinfm": "microsoft/BiomedVLP-CXR-BERT-general"}


@torch.no_grad()
def main(ds):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NAME[ds], trust_remote_code=True)
    m = AutoModel.from_pretrained(NAME[ds], trust_remote_code=True).to(dev).eval()
    ann = D.load_ann(ds)
    out = {}
    for split in ("train", "val", "test"):
        for e in ann[split]:
            cl = D.clauses(ds, e["report"])
            if not cl:
                out[e["id"]] = {"emb": torch.zeros(0, 768, dtype=torch.float16)}
                continue
            b = tok(cl, padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
            h = m(**b).last_hidden_state[:, 0]
            out[e["id"]] = {"emb": torch.nn.functional.normalize(h.float(), dim=-1).half().cpu()}
        print(split, len(ann[split]), flush=True)
    torch.save(out, paths.work(f"clause_emb_{ds}.pt"))
    print("done", sum(v["emb"].shape[0] for v in out.values()), "clauses")


if __name__ == "__main__":
    main(sys.argv[1])
