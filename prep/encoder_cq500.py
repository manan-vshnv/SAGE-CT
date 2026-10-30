"""ResNet-101 slice encoder fine-tuned on CQ500 haemorrhage classification (14 radiologist labels, majority of 3
reads), the PCRL / MEPNet recipe. It stays frozen afterwards. No left-right flip augmentation.

  python -m prep.encoder_cq500          -> work/encoder_cq500.pt
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

from sagect import paths

RES = 512
MAX_SLICES = 32
CQ_DIR = paths.work("slices_cq500")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def cq500_cases():
    labels = json.load(open(f"{CQ_DIR}/labels.json"))
    ids = sorted(int(i) for i in labels if os.path.exists(f"{CQ_DIR}/{i}/done"))
    perm = np.random.RandomState(0).permutation(ids)
    cut = int(0.8 * len(perm))
    mk = lambda i: {"id": f"cq{i}", "labels": {k for k, v in labels[str(i)].items() if v}}  # noqa: E731
    return {"train": [mk(i) for i in perm[:cut]], "val": [mk(i) for i in perm[cut:]]}


def files_of(cid):
    return sorted(f"{CQ_DIR}/{cid[2:]}/{f}" for f in os.listdir(f"{CQ_DIR}/{cid[2:]}") if f.endswith(".jpg"))


def load_img(p):
    im = Image.open(p).convert("L")
    if im.size != (RES, RES):
        im = im.resize((RES, RES), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32)[None] / 255.0
    return (np.repeat(x, 3, 0) - MEAN) / STD


class Studies(Dataset):
    def __init__(self, cases, vocab, label_fn, train):
        self.cases, self.vocab, self.label_fn, self.train = cases, vocab, label_fn, train

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, k):
        c = self.cases[k]
        fs = files_of(c["id"])
        if len(fs) > MAX_SLICES:
            idx = (np.sort(np.random.choice(len(fs), MAX_SLICES, replace=False)) if self.train
                   else np.linspace(0, len(fs) - 1, MAX_SLICES).round().astype(int))
            fs = [fs[i] for i in idx]
        x = np.stack([load_img(p) for p in fs])
        y = np.zeros(len(self.vocab), np.float32)
        for l in self.label_fn(c):
            if l in self.vocab:
                y[self.vocab[l]] = 1
        return torch.from_numpy(x), torch.from_numpy(y)


def cq_labels(case):  # module-level (not a lambda) so DataLoader workers also start on Windows
    return case["labels"]


def collate(batch):
    xs, ys = zip(*batch)
    return list(xs), torch.stack(ys)


class Net(nn.Module):
    """Per-slice ResNet101 -> attention-MIL over slices -> multi-label head."""

    def __init__(self, n_out):
        super().__init__()
        self.trunk = AutoModel.from_pretrained("microsoft/resnet-101")
        self.att = nn.Sequential(nn.Linear(2048, 256), nn.Tanh(), nn.Linear(256, 1))
        self.head = nn.Linear(2048, n_out)

    def fmap(self, x):
        return self.trunk(pixel_values=x).last_hidden_state  # N,2048,16,16 at 512x512

    def forward(self, xs):
        n = [len(x) for x in xs]
        v = self.fmap(torch.cat(xs)).mean((2, 3)).float()
        out = []
        for chunk in torch.split(v, n):
            a = torch.softmax(self.att(chunk), 0)
            out.append(self.head((a * chunk).sum(0)))
        return torch.stack(out)


@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    P, Y = [], []
    for xs, y in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            P.append(torch.sigmoid(model([x.to(dev) for x in xs]).float()).cpu().numpy())
        Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    aucs = {j: roc_auc_score(Y[:, j], P[:, j]) for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)}
    return float(np.mean(list(aucs.values()))), aucs


def train(data, vocab, label_fn, ckpt, epochs=12, lr_trunk=3e-5, batch=2):
    dev = "cuda"
    model = Net(len(vocab)).to(dev)
    tr = DataLoader(Studies(data["train"], vocab, label_fn, True), batch_size=batch, shuffle=True, num_workers=10,
                    collate_fn=collate, persistent_workers=True)
    va = DataLoader(Studies(data["val"], vocab, label_fn, False), batch_size=4, num_workers=10, collate_fn=collate)
    pos = np.zeros(len(vocab))
    for c in data["train"]:
        for l in label_fn(c):
            if l in vocab:
                pos[vocab[l]] += 1
    pw = torch.tensor(np.clip((len(data["train"]) - pos) / np.maximum(pos, 1), 1, 20), dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW([{"params": model.trunk.parameters(), "lr": lr_trunk},
                             {"params": list(model.att.parameters()) + list(model.head.parameters()), "lr": 10 * lr_trunk}],
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_trunk, 10 * lr_trunk], total_steps=epochs * len(tr),
                                                pct_start=0.1)
    best = -1
    inv = {j: l for l, j in vocab.items()}
    for ep in range(epochs):
        model.train()
        for step, (xs, y) in enumerate(tr):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model([x.to(dev, non_blocking=True) for x in xs])
            loss = F.binary_cross_entropy_with_logits(logits.float(), y.to(dev), pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 200 == 0:
                print(f"ep {ep} step {step}/{len(tr)} loss {loss.item():.4f}", flush=True)
        auc, per = evaluate(model, va, dev)
        print(f"ep {ep} val macro-AUROC {auc:.4f} ({len(per)} labels)", flush=True)
        if auc > best:
            best = auc
            torch.save(model.state_dict(), ckpt)
            json.dump({inv[j]: round(a, 4) for j, a in per.items()}, open(ckpt.replace(".pt", "_val_auroc.json"), "w"),
                      indent=1)
    print(f"best val macro-AUROC {best:.4f} -> {ckpt}", flush=True)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    data = cq500_cases()
    vocab = {l: j for j, l in enumerate(json.load(open(f"{CQ_DIR}/labels.json"))["0"])}
    print(f"CQ500 train {len(data['train'])} val {len(data['val'])}, labels {list(vocab)}", flush=True)
    train(data, vocab, cq_labels, paths.work("encoder_cq500.pt"), epochs=20, lr_trunk=5e-5)


if __name__ == "__main__":
    main()
