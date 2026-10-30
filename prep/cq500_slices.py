"""CQ500 DICOM -> CTRG-style axial slices (same processing as indinfm_slices.slabs_from_ras), plus scan labels.

Series choice per scan: a ~5 mm plain series if present, else a thin plain series; bone / contrast series are
skipped. DICOMs are read straight from the zip. Labels = majority vote of the 3 radiologist reads in reads.csv.

  python -m prep.cq500_slices <task_index> <n_tasks>      (CPU only; e.g. 0 1 for a single process)
  python -m prep.cq500_slices labels                      (writes labels.json)
"""
import csv
import io
import json
import os
import re
import sys
import zipfile

import numpy as np
import pydicom

from prep.indinfm_slices import slabs_from_ras
from sagect import paths

ROOT = paths.CQ500_DIR
OUT = paths.work("slices_cq500")
LABELS = ["ICH", "IPH", "IVH", "SDH", "EDH", "SAH", "BleedLocation-Left", "BleedLocation-Right", "ChronicBleed",
          "Fracture", "CalvarialFracture", "OtherFracture", "MassEffect", "MidlineShift"]


def series_score(name):
    n = name.lower().replace("pre contrast", "plain")  # "PRE CONTRAST" = non-contrast scan
    if re.search(r"bone|post|contrast|\bce\b|c\+|angio|cta", n):
        return -1
    if "plain" not in n:
        return 0
    return 3 if re.search(r"5\s*mm|55mm|5\.0", n) else 2  # prefer ~5 mm, like CTRG


def load_series(zf, files):
    dss = []
    for f in files:
        try:
            dss.append(pydicom.dcmread(io.BytesIO(zf.read(f))))
        except Exception:  # non-image entries
            continue
    dss = [d for d in dss if hasattr(d, "ImagePositionPatient") and hasattr(d, "pixel_array")]
    dss.sort(key=lambda d: float(d.ImagePositionPatient[2]))  # LPS z: inferior -> superior
    iop = np.array(dss[0].ImageOrientationPatient, dtype=float).round()
    vol = np.stack([d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1))
                    + float(getattr(d, "RescaleIntercept", 0)) for d in dss], -1)  # rows, cols, z
    # rows run along iop[3:] (posterior for +1), cols along iop[:3] (patient left for +1). Make RAS x,y.
    if iop[0] > 0:
        vol = vol[:, ::-1]  # cols -> towards patient right
    if iop[4] > 0:
        vol = vol[::-1]  # rows -> towards anterior
    hu = np.transpose(vol, (1, 0, 2))  # x (R), y (A), z (S)
    sy, sx = (float(v) for v in dss[0].PixelSpacing)
    zs = [float(d.ImagePositionPatient[2]) for d in dss]
    sz = float(np.median(np.diff(zs))) if len(zs) > 1 else float(getattr(dss[0], "SliceThickness", 5))
    return np.ascontiguousarray(hu), sx, sy, sz


def convert(i):
    zf = zipfile.ZipFile(f"{ROOT}/zips/CQ500-CT-{i}.zip")
    groups = {}
    for f in zf.namelist():
        if f.endswith("/"):
            continue
        groups.setdefault(os.path.dirname(f), []).append(f)
    best = max(groups, key=lambda g: (series_score(os.path.basename(g)), len(groups[g])))
    if series_score(os.path.basename(best)) < 0:
        return None, "no_plain_series"
    hu, sx, sy, sz = load_series(zf, groups[best])
    slices = slabs_from_ras(hu, sx, sy, sz)
    return (slices, os.path.basename(best)) if slices else (None, "no_head")


def labels():
    rows = list(csv.DictReader(open(f"{ROOT}/reads.csv")))
    out = {}
    for r in rows:
        i = int(re.findall(r"\d+", r["name"])[-1])
        out[i] = {l: int(sum(int(r[f"R{k}:{l}"]) for k in (1, 2, 3)) >= 2) for l in LABELS}
    json.dump(out, open(f"{OUT}/labels.json", "w"), indent=1)
    prev = {l: sum(v[l] for v in out.values()) for l in LABELS}
    print(len(out), "scans; positives per label:", prev)


def main():
    task, n = int(sys.argv[1]), int(sys.argv[2])
    log = []
    for i in list(range(491))[task::n]:
        d = f"{OUT}/{i}"
        if os.path.exists(f"{d}/done"):
            continue
        try:
            slices, info = convert(i)
        except Exception as e:  # corrupt or incomplete archive: record and move on
            slices, info = None, f"error: {e}"
        if slices is None:
            log.append({"id": i, "status": info})
            continue
        os.makedirs(d, exist_ok=True)
        for k, im in enumerate(slices):
            im.save(f"{d}/{k:04d}.jpg", quality=95)
        open(f"{d}/done", "w").write(info)
        log.append({"id": i, "status": "ok", "n_slices": len(slices), "series": info})
        print(i, len(slices), info, flush=True)
    os.makedirs(f"{OUT}/_logs", exist_ok=True)
    json.dump(log, open(f"{OUT}/_logs/task{task}.json", "w"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    labels() if sys.argv[1] == "labels" else main()
