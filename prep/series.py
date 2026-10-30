"""
Pick the right CT series for a patient.

Taking `sorted(glob(...))[0]` is not safe here. 15.7% of patients have more
than one Smooth series, and the cohort contains whole-body trauma pan-scans
(`ct-stnd-thin`), lung series, facial-bone series, coronal/MPR reformats and
"processed images" alongside the actual head CT; a pan-scan can sort ahead of
the head series.

Selection combines two cheap signals:
  * the series description embedded in the filename, and
  * the S-I extent read from the NIfTI header (no voxel data is loaded).

A dedicated head CT spans roughly 120-300 mm. Anything much longer is a
pan-scan, where the head is a small fraction of the volume and the vertex may
not even be included.
"""
import glob
import os
import re

import nibabel as nib

HEAD_MIN_MM, HEAD_MAX_MM = 110.0, 330.0

# filename token -> score
_POSITIVE = {"brain": 4.0, "head": 4.0, "plain": 1.5, "soft": 1.0, "stnd": -6.0}
_NEGATIVE = {
    "lung": -8.0, "chest": -8.0, "abdomen": -8.0, "abdo": -8.0, "pelvis": -8.0,
    "whole": -8.0, "body": -6.0, "angio": -3.0, "cta": -3.0,
    "facial": -5.0, "orbit": -4.0, "temporal": -4.0, "petrous": -4.0,
    "sinus": -4.0, "neck": -5.0, "spine": -6.0, "cervical": -5.0,
    "cor": -3.0, "sag": -3.0, "mpr": -3.0, "processed": -4.0, "scout": -9.0,
    "topogram": -9.0, "localizer": -9.0, "dose": -2.0, "report": -9.0,
}


def _describe(path):
    name = os.path.basename(path)
    name = re.sub(r"^[0-9a-f\-]{36}-", "", name)          # strip the uuid
    return re.sub(r"\.nii(\.gz)?$", "", name).lower()


def _si_extent_mm(path):
    """S-I extent in mm, from the header only - no voxel data is read."""
    try:
        img = nib.load(path)
        codes = nib.aff2axcodes(img.affine)
        axis = next((i for i, c in enumerate(codes) if c in ("S", "I")), 2)
        return float(img.shape[axis]) * float(img.header.get_zooms()[axis])
    except Exception:
        return None


def score_series(path, prefer_soft=True):
    desc = _describe(path)
    score = 0.0
    for tok, w in _POSITIVE.items():
        if tok in desc:
            score += w
    for tok, w in _NEGATIVE.items():
        if re.search(rf"(^|[-_]){tok}", desc):
            score += w
    # Bone-kernel series are reconstructed for bone detail; the soft/standard
    # kernel is the better source for the brain window. We still derive a bone
    # WINDOW from whichever volume is chosen.
    if "bone" in desc:
        score += -2.0 if prefer_soft else 2.0

    ext = _si_extent_mm(path)
    if ext is not None:
        if HEAD_MIN_MM <= ext <= HEAD_MAX_MM:
            score += 3.0
        elif ext > HEAD_MAX_MM:
            # Pan-scan: the head is a small part of it and the vertex may be
            # missing entirely.
            score -= 6.0
        else:
            score -= 2.0                                   # too short to be a head
    return score, desc, ext


def find_scan(patient_dir, kernels=("Smooth", "Sharp"), return_all=False):
    """Return the best CT series path for a patient, or None."""
    cands = []
    for k in kernels:
        cands += sorted(glob.glob(os.path.join(patient_dir, "ct", k, "*.nii*")))
    if not cands:
        cands = sorted(glob.glob(os.path.join(patient_dir, "**", "*.nii*"), recursive=True))
    if not cands:
        return ([], None) if return_all else None

    scored = []
    for p in cands:
        s, desc, ext = score_series(p)
        # Prefer Smooth over Sharp when otherwise equal.
        if os.path.basename(os.path.dirname(p)) == "Smooth":
            s += 0.5
        scored.append((s, p, desc, ext))
    scored.sort(key=lambda x: (-x[0], x[1]))
    best = scored[0][1]
    return (scored, best) if return_all else best
