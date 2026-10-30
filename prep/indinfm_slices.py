"""Convert IndiNFM head-CT volumes into CTRG-style axial slices (the source data are only read).

Per patient: pick the head series (prep/series.py:find_scan), reorient to RAS,
keep up to 160 mm below the vertex (superior-anchored, drops neck / pan-scan body), average into 5 mm axial
slabs, crop a square around the head, resize to 512x512, brain window (L40/W80), radiological convention
(patient right on image left, anterior up), ordered inferior -> superior like CTRG. Saved as JPEG.

  python -m prep.indinfm_slices <task_index> <n_tasks>     (CPU only; e.g. 0 1 for a single process)
"""
import json
import os
import sys

import nibabel as nib
import numpy as np
from PIL import Image

from prep.series import find_scan  # noqa: E402  (head-series selection)
from sagect import paths  # noqa: E402

ROOT = os.path.dirname(paths.INDINFM_DIR.rstrip("/"))  # export paths in the cohort json start with its folder name
OUT = paths.work("slices_indinfm")
SLAB_MM, KEEP_MM = 5.0, 160.0


def patients():
    out = []
    for split in ("train", "test"):
        d = json.load(open(f"{paths.INDINFM_DIR}/{split}.json", encoding="utf-8"))
        for inst in d.values():
            for p in inst.values():
                out.append((split, f"{p['institute_id']}_{p['patient_id']}", f"{ROOT}/{p['export_path']}"))
    return out


def convert(pdir):
    path = find_scan(pdir)
    if not path:
        return None, "no_series"
    img = nib.as_closest_canonical(nib.load(path))
    hu = np.asarray(img.dataobj, dtype=np.float32)  # x: ->Right, y: ->Anterior, z: ->Superior
    sx, sy, sz = img.header.get_zooms()[:3]
    slices = slabs_from_ras(hu, sx, sy, sz)
    return (slices, os.path.relpath(path, ROOT)) if slices else (None, "no_head")


def slabs_from_ras(hu, sx, sy, sz):
    """RAS-ordered HU volume -> list of 512x512 brain-window PIL slices (5 mm, inferior -> superior)."""
    head = hu > -300
    zs = np.where(head.reshape(-1, head.shape[2]).sum(0) > 0.01 * head.shape[0] * head.shape[1])[0]
    if len(zs) == 0:
        return None
    top = zs.max()
    bot = max(zs.min(), int(top - KEEP_MM / sz))
    hu, head = hu[:, :, bot: top + 1], head[:, :, bot: top + 1]
    xs = np.where(head.any((1, 2)))[0]
    ys = np.where(head.any((0, 2)))[0]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = 0.55 * max((xs.max() - xs.min()) * sx, (ys.max() - ys.min()) * sy)
    x0, x1 = int(cx - half / sx), int(cx + half / sx)
    y0, y1 = int(cy - half / sy), int(cy + half / sy)
    pad = ((max(0, -x0), max(0, x1 - hu.shape[0])), (max(0, -y0), max(0, y1 - hu.shape[1])), (0, 0))
    hu = np.pad(hu, pad, constant_values=-1024)
    x0, x1, y0, y1 = x0 + pad[0][0], x1 + pad[0][0], y0 + pad[1][0], y1 + pad[1][0]
    hu = hu[x0:x1, y0:y1]
    per = max(1, int(round(SLAB_MM / sz)))
    slices = []
    for k in range(0, hu.shape[2] - per + 1, per):
        slab = hu[:, :, k: k + per].mean(2)
        w = np.clip((slab - 0.0) / 80.0, 0, 1)  # brain window L40/W80
        im = (w.T[::-1, ::-1] * 255).astype(np.uint8)  # rows: anterior->posterior; cols: patient right->left
        slices.append(Image.fromarray(im).resize((512, 512), Image.BILINEAR))
    return slices


def main():
    task, n = int(sys.argv[1]), int(sys.argv[2])
    todo = patients()[task::n]
    log = []
    for split, key, pdir in todo:
        d = f"{OUT}/{key}"
        if os.path.exists(f"{d}/done"):
            continue
        try:
            slices, info = convert(pdir)
        except Exception as e:  # corrupt / unreadable volume: record and move on
            slices, info = None, f"error: {e}"
        if slices is None:
            log.append({"key": key, "split": split, "status": info})
            continue
        os.makedirs(d, exist_ok=True)
        for i, im in enumerate(slices):
            im.save(f"{d}/{i:04d}.jpg", quality=95)
        open(f"{d}/done", "w").write(info)
        log.append({"key": key, "split": split, "status": "ok", "n_slices": len(slices), "series": info})
        print(key, len(slices), flush=True)
    os.makedirs(f"{OUT}/_logs", exist_ok=True)
    json.dump(log, open(f"{OUT}/_logs/task{task}.json", "w"))


if __name__ == "__main__":
    main()
