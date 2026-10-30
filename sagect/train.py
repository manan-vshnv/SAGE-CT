"""Train and test SAGE-CT on one dataset.

  Stage 1  sigmoid contrastive alignment of the pooled latents with the report embedding (checkpoint: lowest
           validation loss); the retrieval index is then built in this space (top-k training studies, never self)
  Stage 2  whole model, label-smoothed cross-entropy, image dropout, one retrieved case replaced by a random one;
           every second epoch the validation reports are scored and the checkpoint with the highest selection score
           is kept, if at least 5% of its validation reports are distinct; greedy decoding; test is scored once.

  python -m sagect.train --ds ctrg --out work/runs/sagect_ctrg_s0 --seed 0
"""
import argparse
import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from . import metrics, paths
from .model import NS, SAGECT

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPLITS = ("train", "val", "test")


class Data:
    def __init__(self, ds, max_len, limit=0, feat="ms4"):
        self.ds, self.L = ds, max_len
        self.ann = D.load_ann(ds)
        if limit:  # smoke test
            self.ann = {s: self.ann[s][:limit] for s in SPLITS}
        cnt = Counter(w for e in self.ann["train"] for w in e["tokens"])
        self.itos = ["<pad>", "<bos>", "<eos>", "<unk>"] + sorted(w for w, n in cnt.items() if n > 3)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        t0 = time.time()
        s4 = lambda i: np.load(paths.work(f"feat_s4_{ds}", f"{i}.npy")).reshape(NS * 16, 2048)  # noqa: E731
        s3 = lambda i: np.load(paths.work(f"feat_s3_{ds}", f"{i}.npy")).reshape(NS * 16, 1024)  # noqa: E731
        ids = [e["id"] for s in SPLITS for e in self.ann[s]]
        if feat == "ms4":  # stage-4 + stage-3 features, 4x4 per slice: 384 tokens x 3072
            self.feat = {i: torch.from_numpy(np.concatenate([s4(i), s3(i)], -1)) for i in ids}
        else:  # "g4", ablation: stage-4 only
            self.feat = {i: torch.from_numpy(s4(i)) for i in ids}
        print(f"features loaded ({len(self.feat)} scans, {time.time() - t0:.0f}s)", flush=True)
        ce = torch.load(paths.work(f"clause_emb_{ds}.pt"))
        self.cl_emb = {k: v["emb"].float() for k, v in ce.items()}
        # report embedding = normalised mean of its clause embeddings
        self.rep_emb = {k: F.normalize(E.mean(0), dim=-1) if len(E) else torch.zeros(768) for k, E in self.cl_emb.items()}

    def encode(self, e):
        return [BOS] + [self.stoi.get(w, UNK) for w in e["tokens"]][: self.L - 1] + [EOS]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS:
                break
            if i in (PAD, BOS):
                continue
            out.append(self.itos[i])
        return D.detok(out, self.ds)


# ================================================================ Stage 1 and retrieval
def run_stage1(model, data, opt):
    params = list(model.res.parameters()) + list(model.scan_head.parameters()) + list(model.text_head.parameters()) + \
        [model.t_logit, model.t_bias]
    o = torch.optim.AdamW(params, lr=opt.lr_a, weight_decay=0.01)
    tr, va = data.ann["train"], data.ann["val"]
    best, best_state = 1e9, None
    for ep in range(opt.epochs_a):
        model.train()
        random.shuffle(tr)
        tot = 0.0
        for i in range(0, len(tr) - opt.bs_a + 1, opt.bs_a):
            rows = tr[i: i + opt.bs_a]
            f = torch.stack([data.feat[e["id"]] for e in rows]).cuda()
            rep = torch.stack([data.rep_emb[e["id"]] for e in rows]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.siglip(model.res(f), rep)
            o.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            o.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            vl = []
            for i in range(0, len(va) - opt.bs_a + 1, opt.bs_a):
                rows = va[i: i + opt.bs_a]
                f = torch.stack([data.feat[e["id"]] for e in rows]).cuda()
                rep = torch.stack([data.rep_emb[e["id"]] for e in rows]).cuda()
                vl.append(model.siglip(model.res(f), rep).item())
            vl = float(np.mean(vl)) if vl else tot
        print(f"stage 1 ep {ep} train {tot:.2f} val {vl:.4f}", flush=True)
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best


@torch.no_grad()
def scan_index(model, data, k, space):
    """top-k training studies of every study (never itself), by cosine of the Stage-1 scan embedding ("warmup") or
    of the mean-pooled frozen input features ("raw", ablation)."""
    model.eval()
    allid = [e["id"] for s in SPLITS for e in data.ann[s]]
    tr = [e["id"] for e in data.ann["train"]]
    if space == "raw":
        S = F.normalize(torch.stack([data.feat[i].float().mean(0) for i in allid]), dim=-1)
    else:
        S = []
        for i in range(0, len(allid), 64):
            f = torch.stack([data.feat[j] for j in allid[i: i + 64]]).cuda()
            S.append(model.scan_emb(model.res(f)).float().cpu())
        S = torch.cat(S)
    pos = {i: j for j, i in enumerate(allid)}
    keys = S[[pos[i] for i in tr]]
    case = {}
    for i in range(0, len(allid), 256):
        for q, row in zip(allid[i: i + 256], (S[i: i + 256] @ keys.T).topk(k + 1, dim=1).indices.tolist()):
            case[q] = [tr[j] for j in row if tr[j] != q][:k]
    return case


def knowledge(model, data, rows, case, train, max_cl=8):
    """knowledge tokens from the clauses (at most max_cl each) of the retrieved cases; training: one case random."""
    tr_ids = [e["id"] for e in data.ann["train"]]
    B, K = len(rows), max(1, max(len(case[e["id"]]) for e in rows)) * max_cl
    E, T, M = torch.zeros(B, K, 768), torch.zeros(B, K, dtype=torch.long), torch.zeros(B, K, dtype=torch.bool)
    for b, e in enumerate(rows):
        nb = list(case[e["id"]])
        if train and nb:
            j, r = random.randrange(len(nb)), random.choice(tr_ids)
            while r == e["id"]:
                r = random.choice(tr_ids)
            nb[j] = r
        p = 0
        for rank, k in enumerate(nb):
            C = data.cl_emb[k][:max_cl]
            E[b, p: p + len(C)], T[b, p: p + len(C)], M[b, p: p + len(C)] = C, rank, True
            p += len(C)
        if not M[b].any():
            M[b, 0] = True
    return model.knowledge(E.cuda(), T.cuda(), M.cuda())


# ================================================================ Stage 2 and decoding
@torch.no_grad()
def generate(model, data, rows, case, bs=48):
    """greedy decoding."""
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        rr = rows[i: i + bs]
        B = len(rr)
        z = model.res(torch.stack([data.feat[e["id"]] for e in rr]).cuda())
        r, rm = knowledge(model, data, rr, case, False)
        y = torch.full((B, 1), BOS, dtype=torch.long, device="cuda")
        done = torch.zeros(B, dtype=torch.bool, device="cuda")
        for _ in range(data.L):
            nxt = model(y, z, r, rm)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            y = torch.cat([y, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        for b, e in enumerate(rr):
            pred = data.decode(y[b, 1:].tolist()) or ("无" if data.ds == "ctrg" else "none")
            out.append({"id": e["id"], "pred": pred, "gt": e["report"]})
    return out


def run_stage2(model, data, case, opt):
    o = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.01)
    tr = data.ann["train"]
    steps = len(tr) // opt.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(o, max_lr=opt.lr, total_steps=steps * opt.epochs, pct_start=0.1)
    best, best_ep, hist = -1e9, -1, []
    for ep in range(opt.epochs):
        model.train()
        random.shuffle(tr)
        tot, t0 = 0.0, time.time()
        for i in range(steps):
            rows = tr[i * opt.bs: (i + 1) * opt.bs]
            B = len(rows)
            seqs = [data.encode(e) for e in rows]
            T = max(map(len, seqs))
            y = torch.tensor([s + [PAD] * (T - len(s)) for s in seqs], device="cuda")
            f = torch.stack([data.feat[e["id"]] for e in rows]).cuda()
            dz = torch.rand(B, device="cuda") < opt.p_drop  # image dropout
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model.res(f)
                r, rm = knowledge(model, data, rows, case, True)
                lg = model(y[:, :-1], z, r, rm, dz)
            loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y[:, 1:].reshape(-1), ignore_index=PAD,
                                   label_smoothing=0.1)
            o.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            o.step()
            sched.step()
            tot += loss.item()
        msg = f"ep {ep} loss {tot / max(steps, 1):.4f} ({time.time() - t0:.0f}s)"
        if (ep + 1) % opt.eval_every == 0 or ep == opt.epochs - 1:
            s = metrics.score_file(generate(model, data, data.ann["val"], case), f"{opt.out}/val_ep{ep}.json", data.ds)
            hist.append({"epoch": ep, **s})
            msg += " | val " + json.dumps(s)
            ok = s.get("distinct_preds", 0) >= opt.min_distinct * len(data.ann["val"])  # no template collapse
            if ok and s["select_score"] > best:
                best, best_ep = s["select_score"], ep
                torch.save(model.state_dict(), f"{opt.out}/model-best.pth")
        print(msg, flush=True)
    return best_ep, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["ctrg", "indinfm"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs_a", type=int, default=20)
    ap.add_argument("--bs_a", type=int, default=64)
    ap.add_argument("--lr_a", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--min_distinct", type=float, default=0.05,
                    help="a checkpoint is eligible only with at least this share of distinct validation reports")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--n_lat", type=int, default=64)
    ap.add_argument("--k", type=int, default=5, help="number of retrieved cases")
    ap.add_argument("--p_drop", type=float, default=0.1, help="image dropout rate")
    # ablations
    ap.add_argument("--feat", default="ms4", choices=["ms4", "g4"], help="g4: stage-4 features only")
    ap.add_argument("--no_rope", action="store_true", help="no slice-row-column rotary code")
    ap.add_argument("--gate", default="evidence", choices=["evidence", "scalar", "state", "nocos"])
    ap.add_argument("--no_rag", action="store_true", help="no retrieved knowledge")
    ap.add_argument("--retr_space", default="warmup", choices=["warmup", "raw"], help="raw: retrieve on CNN features")
    ap.add_argument("--eval_only", action="store_true", help="re-score the test set of an existing run")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N studies per split")
    opt = ap.parse_args()
    random.seed(opt.seed), np.random.seed(opt.seed), torch.manual_seed(opt.seed)
    os.makedirs(opt.out, exist_ok=True)
    data = Data(opt.ds, 100 if opt.ds == "ctrg" else 300, opt.limit, opt.feat)
    in_dim = 3072 if opt.feat == "ms4" else 2048
    model = SAGECT(len(data.itos), data.L, layers=opt.layers, gate=opt.gate, in_dim=in_dim, grid=4,
                   n_rank=max(6, opt.k + 1), n_lat=opt.n_lat).cuda()
    model.no_rag = opt.no_rag
    if opt.no_rope:  # zero rotation = identity
        model.res.ang.zero_()
    if opt.eval_only:
        model.load_state_dict(torch.load(f"{opt.out}/model-best.pth"))
        case = json.load(open(f"{opt.out}/retrieval.json"))
        print("TEST", json.dumps(metrics.score_file(generate(model, data, data.ann["test"], case),
                                                    f"{opt.out}/pred_test.json", data.ds)), flush=True)
        return
    res = {"opts": vars(opt), "params": sum(p.numel() for p in model.parameters())}
    print("trainable parameters", res["params"], flush=True)
    res["stage1_val_loss"] = run_stage1(model, data, opt)
    case = scan_index(model, data, opt.k, opt.retr_space)
    json.dump(case, open(f"{opt.out}/retrieval.json", "w"))
    res["best_epoch"], res["history"] = run_stage2(model, data, case, opt)
    assert os.path.exists(f"{opt.out}/model-best.pth"), "no eligible (non-collapsed) checkpoint"
    model.load_state_dict(torch.load(f"{opt.out}/model-best.pth"))
    t0 = time.time()
    rows = generate(model, data, data.ann["test"], case)
    res["sec_per_scan"] = round((time.time() - t0) / len(rows), 4)
    res["test"] = metrics.score_file(rows, f"{opt.out}/pred_test.json", data.ds)
    print("TEST", json.dumps(res["test"]), flush=True)
    json.dump(res, open(f"{opt.out}/result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
