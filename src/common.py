"""Shared loaders, geometry and metric primitives for CycleQA.

Everything works at the stored 512x512 resolution. Dice is scale-invariant, so it
is computed directly. Volumes are corrected back to physical units by rescaling the
in-plane spacing (pixdim is given at the ORIGINAL resolution).
"""
import collections
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLS = {1: "RV", 2: "Myo", 3: "LV"}
LV = 3

DATASETS = {
    "ACDC": "preprocessed",
    "MnM": "preprocessed_mnm",
}

# model -> {dataset: (result_dir, key)}.  A key of None means the model does not
# produce that prediction (frame-wise models have no propagation direction).
MODELS = {
    "MedSAM2-dual": {"ACDC": ("results/medsam2", "bidir"), "MnM": ("results/medsam2_mnm", "bidir")},
    "MedSAM2-ED":   {"ACDC": ("results/medsam2", "ed_pred"), "MnM": ("results/medsam2_mnm", "ed_pred")},
    "MedSAM2-ES":   {"ACDC": ("results/medsam2", "es_pred"), "MnM": ("results/medsam2_mnm", "es_pred")},
    "SAM2-dual":    {"ACDC": ("results/sam2_bidir", "bidir"), "MnM": ("results/sam2_bidir_mnm", "bidir")},
    # Provenance of the checkpoint behind these two is unresolved (infer_unet_mnm.py
    # was deleted in commit c437e26). Reported with that caveat; not in the primary table.
    "U-Net":        {"ACDC": (None, None), "MnM": ("results/unet_mnm", "bidir")},
    "DINOv2":       {"ACDC": (None, None), "MnM": ("results/dinov2_mnm", "bidir")},
}

# Models with two independent propagation directions -> CycleQA disagreement is defined.
DUAL_ANCHORED = {
    "MedSAM2-dual": {"ACDC": "results/medsam2", "MnM": "results/medsam2_mnm"},
    "SAM2-dual":    {"ACDC": "results/sam2_bidir", "MnM": "results/sam2_bidir_mnm"},
}


def dice(a, b, c):
    p, g = (a == c), (b == c)
    s = p.sum() + g.sum()
    if s == 0:
        return np.nan
    return 2.0 * (p & g).sum() / s


def patient_slices(dataset):
    d = collections.defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, DATASETS[dataset], "*.npz")):
        d[os.path.basename(f).split("_slice")[0]].append(f)
    return {k: sorted(v) for k, v in d.items()}


def voxel_mm3_at_512(pixdim, orig_H, orig_W):
    """Physical volume of one voxel after the 512x512 resize."""
    return (float(pixdim[0]) * orig_H / 512.0
            * float(pixdim[1]) * orig_W / 512.0
            * float(pixdim[2]))


def lv_volume_curve(pred_stack, vox_mm3):
    """pred_stack: (T, Z, 512, 512) -> V(t) in mL."""
    return (pred_stack == LV).sum(axis=(1, 2, 3)) * vox_mm3 / 1000.0


def ef_edv_esv(vol_curve, ed_idx, es_idx):
    """EDV is the larger of the two annotated-frame volumes: robust to vendors whose
    ED/ES frame order differs. Mirrors compute_all_metrics.py:235."""
    T = len(vol_curve)
    v_ed = float(vol_curve[min(ed_idx, T - 1)])
    v_es = float(vol_curve[min(es_idx, T - 1)])
    edv, esv = max(v_ed, v_es), min(v_ed, v_es)
    ef = (edv - esv) / edv * 100.0 if edv > 1e-3 else np.nan
    return ef, edv, esv


def load_patient(dataset, pid, files, result_dir, key):
    """Returns dict with GT masks, the prediction stack, and geometry. None if absent."""
    d0 = np.load(files[0], allow_pickle=True)
    ed_idx, es_idx = int(d0["ed_idx"]), int(d0["es_idx"])
    orig_H, orig_W = int(d0["orig_H"]), int(d0["orig_W"])
    vox = voxel_mm3_at_512(d0["pixdim"], orig_H, orig_W)

    gt_ed, gt_es, preds = [], [], []
    for f in files:
        rp = os.path.join(ROOT, result_dir, os.path.basename(f))
        if not os.path.exists(rp):
            return None
        r = np.load(rp, allow_pickle=True)
        if key not in r.files:
            return None
        dd = np.load(f, allow_pickle=True)
        gt_ed.append(dd["ed_mask"])
        gt_es.append(dd["es_mask"])
        preds.append(r[key])

    T = min(p.shape[0] for p in preds)
    return {
        "gt_ed": np.stack(gt_ed),                              # (Z,512,512)
        "gt_es": np.stack(gt_es),
        "pred": np.stack([p[:T] for p in preds], axis=1),      # (T,Z,512,512)
        "ed_idx": ed_idx, "es_idx": es_idx, "T": T,
        "vox_mm3": vox, "group": str(d0["group"]),
    }


def mnm_metadata():
    df = pd.read_csv(os.path.join(ROOT, "MnM", "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"))
    return df.set_index("External code")[["VendorName", "Centre", "Pathology"]]


def acdc_pathology(group):
    """ACDC 'group' field already carries the pathology (NOR/DCM/HCM/MINF/RV)."""
    return group.strip().strip("b'").strip("'")


def patient_context(dataset, pid, group, meta):
    if dataset == "MnM" and pid in meta.index:
        r = meta.loc[pid]
        return {"vendor": r.VendorName, "centre": int(r.Centre), "pathology": r.Pathology}
    return {"vendor": "Siemens", "centre": 0, "pathology": acdc_pathology(group)}
