"""Frozen CQ500 ResNet-101 features of the 24 slices of every study, pooled to a 4x4 grid per slice:
  stage 4 (2048-d, 16x16 at 512x512) -> work/feat_s4_<ds>/<id>.npy  (24, 4, 4, 2048) float16
  stage 3 (1024-d, 32x32)            -> work/feat_s3_<ds>/<id>.npy  (24, 4, 4, 1024) float16
CTRG-Brain uses the 24 slices listed in PCRL's split; IndiNFM 24 slices sampled evenly from the converted series.

  python -m prep.features ctrg [shard] [n_shards]       (GPU)
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from prep.encoder_cq500 import load_img
from sagect import data as D
from sagect import paths


def slice_paths(ds, case):
    if ds == "ctrg":
        return [f"{paths.CTRG_DIR}/{case['id']}/{os.path.basename(p)}" for p in case["images"]]
    fs = sorted(glob.glob(paths.work("slices_indinfm", case["id"], "*.jpg")))
    return [fs[i] for i in np.linspace(0, len(fs) - 1, 24).round().astype(int)]


def first(batch):  # module-level (not a lambda) so DataLoader workers also start on Windows
    return batch[0]


class Cases(Dataset):
    def __init__(self, ds, cases):
        self.ds, self.cases = ds, cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, k):
        c = self.cases[k]
        return c["id"], torch.from_numpy(np.stack([load_img(p) for p in slice_paths(self.ds, c)]))


@torch.no_grad()
def main(ds, shard=0, n_shards=1):
    from transformers import AutoModel
    o4, o3 = paths.work(f"feat_s4_{ds}"), paths.work(f"feat_s3_{ds}")
    os.makedirs(o4, exist_ok=True), os.makedirs(o3, exist_ok=True)
    trunk = AutoModel.from_pretrained("microsoft/resnet-101").cuda().eval()
    st = torch.load(paths.work("encoder_cq500.pt"), map_location="cuda")
    trunk.load_state_dict({k[6:]: v for k, v in st.items() if k.startswith("trunk.")})
    cases = [c for s in ("train", "validation", "test") for c in D.load_split(ds, s)][shard::n_shards]
    cases = [c for c in cases if not os.path.exists(f"{o3}/{c['id']}.npy")]
    dl = DataLoader(Cases(ds, cases), batch_size=1, num_workers=7, collate_fn=first)
    for n, (cid, x) in enumerate(dl):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = trunk(pixel_values=x.cuda(), output_hidden_states=True)
        s4 = out.last_hidden_state.float().half().float()  # (24, 2048, 16, 16), stored as float16 before pooling
        s3 = [h for h in out.hidden_states if h.shape[1] == 1024][-1].float()  # (24, 1024, 32, 32): last stage-3 map
        for o, g in ((o4, s4), (o3, s3)):
            g = F.adaptive_avg_pool2d(g, 4)  # (24, C, 4, 4)
            tmp = f"{o}/{cid}.tmp.npy"
            np.save(tmp, g.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16))
            os.replace(tmp, f"{o}/{cid}.npy")
        if n % 500 == 0:
            print(f"{ds} shard {shard}: {n}/{len(cases)}", flush=True)
    print(ds, "done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], *map(int, sys.argv[2:4]))
