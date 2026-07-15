"""Phase 1 -- CycleQA no-reference features.

HARD RULE: every feature here is computable from the predicted masks alone, at a site
with no ground truth. No reference mask, no true Dice, no reference EF, no vendor,
centre, or pathology label may enter the matrix. run() asserts this before writing.

Four families:
  ANCHOR  -- bidirectional propagation disagreement (THE primary signal). ed_pred
             is the ED-anchored propagation; es_pred is the ES-anchored
             propagation (each anchor propagates both ways to span every frame). Both cover all T frames. If the model is tracking the true anatomy the
             two independent trajectories must converge; if it is drifting they
             separate. Defined only for dual-anchored propagation models -- a frame-wise
             CNN has no second direction and cannot produce this signal at all.
  ANAT    -- anatomical plausibility: the myocardium must form a closed ring enclosing
             the LV pool; no fragments, no holes, no image-border contact.
  TEMP    -- temporal coherence: V(t) must be smooth and close its cycle.
  FUNC    -- physiological plausibility: EDV > ESV, EF in a possible range.

Speed: masks are streamed one slice at a time and reduced on the fly. A 3D Dice or IoU
is a ratio of summed voxel counts, so it accumulates exactly across slices -- the
(T, Z, 512, 512) stack is never allocated. Each slice is additionally cropped to its own
bounding box (the heart fills a small part of the frame). Peak memory per worker is
~25 MB rather than ~250 MB, which is what made the stacked version thrash.

Output: metis/results/patient_features.csv        (MedSAM2-dual, primary)
        metis/results/patient_features_sam2.csv   (a second dual-anchored FM)
        metis/results/disagreement_curves.npz     (dense A(t) for the qualitative figure)
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion, label as cc_label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CLS, DUAL_ANCHORED, LV, ROOT, ef_edv_esv, patient_slices,
                    voxel_mm3_at_512)

FORBIDDEN = ("gt_", "true_", "dice_", "hd95", "assd", "vendor", "centre",
             "pathology", "ef_abs_err", "patient_id")
NC = len(CLS)


def one(args):
    dataset, pid, files, rdir = args
    d0 = np.load(files[0], allow_pickle=True)
    ed_idx, es_idx = int(d0["ed_idx"]), int(d0["es_idx"])
    vox = voxel_mm3_at_512(d0["pixdim"], int(d0["orig_H"]), int(d0["orig_W"]))

    T = None
    acc = None          # lazily sized once T is known
    anat = {"ncc": [], "contain": [], "ring": [], "holes": []}
    cent_shift = []
    border = []

    for f in files:
        rp = os.path.join(ROOT, rdir, os.path.basename(f))
        if not os.path.exists(rp):
            return None
        r = np.load(rp, allow_pickle=True)
        if not {"ed_pred", "es_pred", "bidir"} <= set(r.files):
            return None
        ed, es, bi = r["ed_pred"], r["es_pred"], r["bidir"]
        t = min(ed.shape[0], es.shape[0], bi.shape[0])
        ed, es, bi = ed[:t], es[:t], bi[:t]

        if T is None:
            T = t
            acc = {
                "a_int": np.zeros((T, NC)), "a_p": np.zeros((T, NC)), "a_g": np.zeros((T, NC)),
                "i_int": np.zeros((T - 1, NC)), "i_uni": np.zeros((T - 1, NC)),
                "lv": np.zeros(T),
            }
        elif t != T:
            T = min(T, t)

        # border contact is a property of the ORIGINAL frame -- measure before cropping
        nzb = bi > 0
        border.append(float(nzb[:, 0, :].any() or nzb[:, -1, :].any()
                            or nzb[:, :, 0].any() or nzb[:, :, -1].any()))

        # crop this slice to its own bounding box (+8 px)
        u = (ed > 0) | (es > 0) | nzb
        ys, xs = np.where(u.any(axis=0))
        if ys.size == 0:
            continue
        y0, y1 = max(0, ys.min() - 8), min(512, ys.max() + 9)
        x0, x1 = max(0, xs.min() - 8), min(512, xs.max() + 9)
        ed, es, bi = (a[:, y0:y1, x0:x1] for a in (ed, es, bi))

        for ci, c in enumerate(CLS):
            pe, ps = (ed == c), (es == c)
            acc["a_int"][:T, ci] += (pe & ps)[:T].sum((1, 2))
            acc["a_p"][:T, ci] += pe[:T].sum((1, 2))
            acc["a_g"][:T, ci] += ps[:T].sum((1, 2))

            b = bi == c
            b0, b1 = b[:-1], b[1:]
            acc["i_int"][:T - 1, ci] += (b0 & b1)[:T - 1].sum((1, 2))
            acc["i_uni"][:T - 1, ci] += (b0 | b1)[:T - 1].sum((1, 2))

        lv = bi == LV
        acc["lv"][:T] += lv[:T].sum((1, 2))

        # per-slice LV centroid displacement between consecutive frames
        for tt in range(T - 1):
            a_, b_ = lv[tt], lv[tt + 1]
            if a_.sum() > 20 and b_.sum() > 20:
                ca = np.array(np.nonzero(a_)).mean(1)
                cb = np.array(np.nonzero(b_)).mean(1)
                cent_shift.append(float(np.linalg.norm(ca - cb)))

        # anatomy on a subsample of frames
        for tt in range(0, T, max(1, T // 12)):
            m = bi[tt]
            if m.sum() == 0:
                continue
            for c in CLS:
                bb = m == c
                if bb.sum() > 10:
                    anat["ncc"].append(cc_label(bb)[1])
            lvm, myo = m == LV, m == 2
            if lvm.sum() > 20:
                rim = binary_dilation(lvm, iterations=2) & ~lvm
                anat["contain"].append((myo & rim).sum() / max(1, rim.sum()))
                anat["ring"].append(1.0 if binary_erosion(myo, iterations=1).sum() > 0 else 0.0)
                anat["holes"].append(float(cc_label(~lvm)[1] > 1))

    if acc is None:
        return None

    # ---- ANCHOR: exact 3D Dice per frame per class, from the accumulated counts
    den = acc["a_p"] + acc["a_g"]
    A = np.divide(2.0 * acc["a_int"], den, out=np.full_like(den, np.nan), where=den > 0)  # (T,NC)

    lo, hi = sorted((ed_idx, es_idx))
    pad = max(1, int(0.25 * max(1, hi - lo)))
    mid = [t for t in range(T) if lo + pad <= t <= hi - pad] or list(range(T))
    anchored = [min(ed_idx, T - 1), min(es_idx, T - 1)]

    row = {"patient_id": pid, "dataset": dataset}
    for ci, name in enumerate(CLS.values()):
        a = A[:, ci]
        fin = a[np.isfinite(a)]
        if fin.size == 0:
            fin = np.array([np.nan])
        row[f"anchor_agree_mean_{name}"] = np.nanmean(a)
        row[f"anchor_agree_min_{name}"] = np.nanmin(fin)
        row[f"anchor_agree_std_{name}"] = np.nanstd(a)
        row[f"anchor_agree_p10_{name}"] = np.nanpercentile(fin, 10)
        row[f"anchor_agree_mid_{name}"] = np.nanmean(a[mid])
        row[f"anchor_frac_below70_{name}"] = float(np.nanmean(a < 0.70))
        row[f"anchor_decay_{name}"] = np.nanmean(a[mid]) - np.nanmean(a[anchored])
    fin = A[np.isfinite(A)]
    row["anchor_agree_mean_ALL"] = np.nanmean(A)
    row["anchor_agree_min_ALL"] = fin.min() if fin.size else np.nan
    row["anchor_agree_mid_ALL"] = np.nanmean(A[mid])

    # ---- ANAT
    row["anat_n_components"] = np.mean(anat["ncc"]) if anat["ncc"] else np.nan
    row["anat_max_components"] = np.max(anat["ncc"]) if anat["ncc"] else np.nan
    row["anat_lv_containment"] = np.mean(anat["contain"]) if anat["contain"] else np.nan
    row["anat_myo_ring_closed"] = np.mean(anat["ring"]) if anat["ring"] else np.nan
    row["anat_lv_holes"] = np.mean(anat["holes"]) if anat["holes"] else np.nan
    row["anat_border_contact"] = float(np.mean(border)) if border else np.nan

    # ---- TEMP
    iou = np.divide(acc["i_int"], acc["i_uni"],
                    out=np.full_like(acc["i_uni"], np.nan), where=acc["i_uni"] > 20)
    fin = iou[np.isfinite(iou)]
    row["temp_interframe_iou"] = np.nanmean(iou)
    row["temp_min_iou"] = fin.min() if fin.size else np.nan
    row["temp_centroid_shift"] = np.mean(cent_shift) if cent_shift else np.nan
    row["temp_max_centroid_shift"] = np.max(cent_shift) if cent_shift else np.nan

    v = acc["lv"][:T] * vox / 1000.0
    mv = max(np.nanmax(v), 1e-6)
    d1 = np.diff(v)
    row["temp_max_area_jump"] = float(np.nanmax(np.abs(d1)) / mv) if len(d1) else np.nan
    row["temp_curve_roughness"] = float(np.nanmean(np.abs(np.diff(v, 2))) / mv) if len(v) > 2 else np.nan
    row["temp_cycle_closure"] = float(abs(v[0] - v[-1]) / mv) if len(v) > 1 else np.nan
    row["temp_n_extrema"] = float((np.diff(np.sign(d1)) != 0).sum()) if len(d1) > 1 else np.nan

    # ---- FUNC
    ef, edv, esv = ef_edv_esv(v, ed_idx, es_idx)
    ok = np.isfinite(ef) and 10.0 <= ef <= 80.0
    row["func_pred_EF"] = ef
    row["func_pred_EDV"] = edv
    row["func_pred_ESV"] = esv
    row["func_edv_exceeds_esv"] = float(edv > esv)
    row["func_ef_implausible"] = float(not ok)
    row["func_vol_cv"] = float(np.nanstd(v) / max(np.nanmean(v), 1e-6))
    row["func_min_vol_frac"] = float(np.nanmin(v) / mv)
    row["func_es_is_argmin"] = float(int(np.nanargmin(v)) == min(es_idx, T - 1))

    return row, f"{dataset}/{pid}", A.T          # A.T: (NC, T)


def run(model, out_name, save_curves=False):
    jobs = []
    for dataset, rdir in DUAL_ANCHORED[model].items():
        for pid, files in patient_slices(dataset).items():
            jobs.append((dataset, pid, files, rdir))

    n = len(jobs)
    nproc = int(os.environ.get("CYCLEQA_PROCS", min(48, os.cpu_count() or 8)))
    print(f"[{model}] {n} patients on {nproc} procs", flush=True)

    res, t0 = [], time.time()
    with Pool(nproc) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=1), 1):
            if r:
                res.append(r)
            if i % 20 == 0 or i == n:
                el = time.time() - t0
                print(f"  [{i:4d}/{n}] {100*i/n:5.1f}%  elapsed {el:6.1f}s  "
                      f"eta {el/i*(n-i):6.1f}s", flush=True)

    df = pd.DataFrame([r[0] for r in res]).sort_values(["dataset", "patient_id"])
    feat = [c for c in df.columns if c not in ("patient_id", "dataset")]
    leaks = [c for c in feat if any(k in c.lower() for k in FORBIDDEN)]
    assert not leaks, f"LEAKAGE: forbidden columns in feature matrix: {leaks}"

    out = os.path.join(ROOT, "metis", "results", out_name)
    df.to_csv(out, index=False)
    print(f"[{model}] wrote {out}  ({len(df)} patients, {len(feat)} features, "
          f"leakage check PASSED)\n", flush=True)

    if save_curves:
        np.savez_compressed(os.path.join(ROOT, "metis", "results", "disagreement_curves.npz"),
                            **{r[1]: r[2] for r in res})
    return df


if __name__ == "__main__":
    run("MedSAM2-dual", "patient_features.csv", save_curves=True)
    run("SAM2-dual", "patient_features_sam2.csv")
    print("[extract_features] done. Frame-wise models (U-Net, DINOv2) are absent by "
          "construction: with no propagation there is no second direction to disagree.")
