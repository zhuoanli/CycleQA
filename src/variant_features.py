"""Phase 2b (part 1) -- per-variant no-reference features, for QC-guided reselection.

Dual-anchored MedSAM2 already stores three predictions per patient: the merged `bidir`
output and the two single-anchor propagations `ed_pred` and `es_pred`. Gate 0b showed
that the best of the three is domain-dependent -- on ACDC the merged output wins, but on
M&Ms the ES-anchored propagation is markedly better (EF MAE 16.8 vs 23.4 pp).

At a new site you cannot use ground truth to pick. But you can score each variant with
the same no-reference plausibility features and take the one CycleQA trusts most. That
is a label-free, retraining-free, zero-extra-inference intervention: all three
predictions are already on disk.

Output: metis/results/variant_features.csv  (one row per patient x variant)
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_erosion, label as cc_label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLS, DUAL_ANCHORED, LV, ROOT, ef_edv_esv, patient_slices, voxel_mm3_at_512

VARIANTS = ["bidir", "ed_pred", "es_pred"]
NC = len(CLS)


def one(args):
    dataset, pid, files, rdir = args
    d0 = np.load(files[0], allow_pickle=True)
    ed_idx, es_idx = int(d0["ed_idx"]), int(d0["es_idx"])
    vox = voxel_mm3_at_512(d0["pixdim"], int(d0["orig_H"]), int(d0["orig_W"]))

    T = None
    acc = {v: None for v in VARIANTS}
    anat = {v: {"ncc": [], "contain": [], "ring": [], "holes": []} for v in VARIANTS}
    cent = {v: [] for v in VARIANTS}
    border = {v: [] for v in VARIANTS}

    for f in files:
        rp = os.path.join(ROOT, rdir, os.path.basename(f))
        if not os.path.exists(rp):
            return None
        r = np.load(rp, allow_pickle=True)
        if not set(VARIANTS) <= set(r.files):
            return None
        seqs = {v: r[v] for v in VARIANTS}
        t = min(s.shape[0] for s in seqs.values())
        T = t if T is None else min(T, t)

        for v in VARIANTS:
            m = seqs[v][:t]
            if acc[v] is None:
                acc[v] = {"i_int": np.zeros((t - 1, NC)), "i_uni": np.zeros((t - 1, NC)),
                          "lv": np.zeros(t)}
            nz = m > 0
            border[v].append(float(nz[:, 0, :].any() or nz[:, -1, :].any()
                                   or nz[:, :, 0].any() or nz[:, :, -1].any()))
            ys, xs = np.where(nz.any(axis=0))
            if ys.size == 0:
                continue
            y0, y1 = max(0, ys.min() - 8), min(512, ys.max() + 9)
            x0, x1 = max(0, xs.min() - 8), min(512, xs.max() + 9)
            m = m[:, y0:y1, x0:x1]

            for ci, c in enumerate(CLS):
                b = m == c
                b0, b1 = b[:-1], b[1:]
                acc[v]["i_int"][:t - 1, ci] += (b0 & b1).sum((1, 2))
                acc[v]["i_uni"][:t - 1, ci] += (b0 | b1).sum((1, 2))
            lv = m == LV
            acc[v]["lv"][:t] += lv.sum((1, 2))

            for tt in range(t - 1):
                a_, b_ = lv[tt], lv[tt + 1]
                if a_.sum() > 20 and b_.sum() > 20:
                    cent[v].append(float(np.linalg.norm(
                        np.array(np.nonzero(a_)).mean(1) - np.array(np.nonzero(b_)).mean(1))))

            for tt in range(0, t, max(1, t // 12)):
                mm = m[tt]
                if mm.sum() == 0:
                    continue
                for c in CLS:
                    bb = mm == c
                    if bb.sum() > 10:
                        anat[v]["ncc"].append(cc_label(bb)[1])
                lvm, myo = mm == LV, mm == 2
                if lvm.sum() > 20:
                    rim = binary_dilation(lvm, iterations=2) & ~lvm
                    anat[v]["contain"].append((myo & rim).sum() / max(1, rim.sum()))
                    anat[v]["ring"].append(1.0 if binary_erosion(myo, iterations=1).sum() > 0 else 0.0)
                    anat[v]["holes"].append(float(cc_label(~lvm)[1] > 1))

    if any(acc[v] is None for v in VARIANTS):
        return None

    rows = []
    for v in VARIANTS:
        a, n = acc[v], anat[v]
        row = {"patient_id": pid, "dataset": dataset, "variant": v}
        row["anat_n_components"] = np.mean(n["ncc"]) if n["ncc"] else np.nan
        row["anat_max_components"] = np.max(n["ncc"]) if n["ncc"] else np.nan
        row["anat_lv_containment"] = np.mean(n["contain"]) if n["contain"] else np.nan
        row["anat_myo_ring_closed"] = np.mean(n["ring"]) if n["ring"] else np.nan
        row["anat_lv_holes"] = np.mean(n["holes"]) if n["holes"] else np.nan
        row["anat_border_contact"] = float(np.mean(border[v])) if border[v] else np.nan

        iou = np.divide(a["i_int"], a["i_uni"],
                        out=np.full_like(a["i_uni"], np.nan), where=a["i_uni"] > 20)
        fin = iou[np.isfinite(iou)]
        row["temp_interframe_iou"] = np.nanmean(iou)
        row["temp_min_iou"] = fin.min() if fin.size else np.nan
        row["temp_centroid_shift"] = np.mean(cent[v]) if cent[v] else np.nan
        row["temp_max_centroid_shift"] = np.max(cent[v]) if cent[v] else np.nan

        vol = a["lv"][:T] * vox / 1000.0
        mv = max(np.nanmax(vol), 1e-6)
        d1 = np.diff(vol)
        row["temp_max_area_jump"] = float(np.nanmax(np.abs(d1)) / mv) if len(d1) else np.nan
        row["temp_curve_roughness"] = float(np.nanmean(np.abs(np.diff(vol, 2))) / mv) if len(vol) > 2 else np.nan
        row["temp_cycle_closure"] = float(abs(vol[0] - vol[-1]) / mv) if len(vol) > 1 else np.nan
        row["temp_n_extrema"] = float((np.diff(np.sign(d1)) != 0).sum()) if len(d1) > 1 else np.nan

        ef, edv, esv = ef_edv_esv(vol, ed_idx, es_idx)
        ok = np.isfinite(ef) and 10.0 <= ef <= 80.0
        row["func_pred_EF"] = ef
        row["func_pred_EDV"] = edv
        row["func_pred_ESV"] = esv
        row["func_edv_exceeds_esv"] = float(edv > esv)
        row["func_ef_implausible"] = float(not ok)
        row["func_vol_cv"] = float(np.nanstd(vol) / max(np.nanmean(vol), 1e-6))
        row["func_min_vol_frac"] = float(np.nanmin(vol) / mv)
        row["func_es_is_argmin"] = float(int(np.nanargmin(vol)) == min(es_idx, T - 1))
        rows.append(row)
    return rows


def main():
    jobs = []
    for dataset, rdir in DUAL_ANCHORED["MedSAM2-dual"].items():
        for pid, files in patient_slices(dataset).items():
            jobs.append((dataset, pid, files, rdir))
    n = len(jobs)
    nproc = int(os.environ.get("CYCLEQA_PROCS", 48))
    print(f"[variant_features] {n} patients x {len(VARIANTS)} variants on {nproc} procs", flush=True)

    out, t0 = [], time.time()
    with Pool(nproc) as pool:
        for i, rr in enumerate(pool.imap_unordered(one, jobs, chunksize=1), 1):
            if rr:
                out.extend(rr)
            if i % 60 == 0 or i == n:
                el = time.time() - t0
                print(f"  [{i:4d}/{n}] {100*i/n:5.1f}%  {el:6.1f}s", flush=True)

    df = pd.DataFrame(out).sort_values(["dataset", "patient_id", "variant"])
    p = os.path.join(ROOT, "metis", "results", "variant_features.csv")
    df.to_csv(p, index=False)
    print(f"[variant_features] wrote {p}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
