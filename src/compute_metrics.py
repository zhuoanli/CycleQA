"""Gate 0b -- per-patient reference metrics. GT is used HERE ONLY.

These define and evaluate the failure endpoint. They never enter the CycleQA feature
matrix (extract_features.py asserts that separately).

Dice is evaluated at the annotated phases only (ED and ES) -- ACDC and M&Ms provide
reference masks nowhere else.

Speed: masks are streamed slice by slice and reduced on the fly. Dice needs only the
two annotated frames, and the LV volume curve is a per-slice voxel count, so the full
(T, Z, 512, 512) stack is never allocated. All prediction keys stored in one result
directory (bidir / ed_pred / es_pred) are computed from a single read of the file.

Output: metis/results/patient_metrics.csv, one row per (patient, model).
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CLS, LV, ROOT, ef_edv_esv, mnm_metadata, patient_context,
                    patient_slices, voxel_mm3_at_512)

META = mnm_metadata()

# result dir -> {model name: npz key}.  One read serves every key it holds.
DIR_MODELS = {
    "ACDC": {
        "results/medsam2":    {"MedSAM2-dual": "bidir", "MedSAM2-ED": "ed_pred", "MedSAM2-ES": "es_pred"},
        "results/sam2_bidir": {"SAM2-dual": "bidir"},
    },
    "MnM": {
        "results/medsam2_mnm":    {"MedSAM2-dual": "bidir", "MedSAM2-ED": "ed_pred", "MedSAM2-ES": "es_pred"},
        "results/sam2_bidir_mnm": {"SAM2-dual": "bidir"},
        # Checkpoint provenance unresolved (infer_unet_mnm.py deleted in c437e26).
        # Reported with that caveat; excluded from the primary table.
        "results/unet_mnm":       {"U-Net": "bidir"},
        "results/dinov2_mnm":     {"DINOv2": "bidir"},
    },
}


def one(args):
    dataset, pid, files, rdir, keymap = args
    d0 = np.load(files[0], allow_pickle=True)
    ed_idx, es_idx = int(d0["ed_idx"]), int(d0["es_idx"])
    vox = voxel_mm3_at_512(d0["pixdim"], int(d0["orig_H"]), int(d0["orig_W"]))
    group = str(d0["group"])

    keys = list(keymap.values())
    # streaming accumulators, per key
    inter = {k: {c: {"ED": 0, "ES": 0} for c in CLS} for k in keys}   # |p & g|
    psum = {k: {c: {"ED": 0, "ES": 0} for c in CLS} for k in keys}    # |p|
    gsum = {c: {"ED": 0, "ES": 0} for c in CLS}                        # |g|
    lvcnt = {k: None for k in keys}                                    # LV voxels per frame
    gt_lv = {"ED": 0, "ES": 0}
    T = None

    for f in files:
        rp = os.path.join(ROOT, rdir, os.path.basename(f))
        if not os.path.exists(rp):
            return []
        r = np.load(rp, allow_pickle=True)
        if any(k not in r.files for k in keys):
            return []
        dd = np.load(f, allow_pickle=True)
        gts = {"ED": dd["ed_mask"], "ES": dd["es_mask"]}

        for c in CLS:
            for ph in ("ED", "ES"):
                gsum[c][ph] += int((gts[ph] == c).sum())
        gt_lv["ED"] += int((gts["ED"] == LV).sum())
        gt_lv["ES"] += int((gts["ES"] == LV).sum())

        for k in keys:
            seq = r[k]                                   # (T,512,512) uint8
            t = seq.shape[0]
            T = t if T is None else min(T, t)
            for ph, idx in (("ED", min(ed_idx, t - 1)), ("ES", min(es_idx, t - 1))):
                fr = seq[idx]
                for c in CLS:
                    p = fr == c
                    psum[k][c][ph] += int(p.sum())
                    inter[k][c][ph] += int((p & (gts[ph] == c)).sum())
            v = (seq == LV).reshape(t, -1).sum(1).astype(np.int64)
            lvcnt[k] = v if lvcnt[k] is None else lvcnt[k][:len(v)] + v[:len(lvcnt[k])]

    ctx = patient_context(dataset, pid, group, META)
    ef_g, edv_g, esv_g = ef_edv_esv(
        np.array([gt_lv["ED"], gt_lv["ES"]], dtype=float) * vox / 1000.0, 0, 1)

    rows = []
    for model, k in keymap.items():
        row = {"patient_id": pid, "dataset": dataset, "model": model, **ctx}
        for c, name in CLS.items():
            ds = {}
            for ph in ("ED", "ES"):
                den = psum[k][c][ph] + gsum[c][ph]
                ds[ph] = 2.0 * inter[k][c][ph] / den if den else np.nan
            row[f"dice_{name}_ED"], row[f"dice_{name}_ES"] = ds["ED"], ds["ES"]
            row[f"dice_{name}"] = np.nanmean([ds["ED"], ds["ES"]])
        row["dice_min"] = np.nanmin([row[f"dice_{n}"] for n in CLS.values()])
        row["dice_mean"] = np.nanmean([row[f"dice_{n}"] for n in CLS.values()])

        vc = lvcnt[k].astype(float) * vox / 1000.0
        ef_p, edv_p, esv_p = ef_edv_esv(vc, ed_idx, es_idx)
        row.update(pred_EF=ef_p, pred_EDV=edv_p, pred_ESV=esv_p,
                   gt_EF=ef_g, gt_EDV=edv_g, gt_ESV=esv_g,
                   ef_abs_err=abs(ef_p - ef_g) if np.isfinite(ef_p) and np.isfinite(ef_g) else np.nan)
        rows.append(row)
    return rows


def main():
    jobs = []
    for dataset in ("ACDC", "MnM"):
        pats = patient_slices(dataset)
        for rdir, keymap in DIR_MODELS[dataset].items():
            for pid, files in pats.items():
                jobs.append((dataset, pid, files, rdir, keymap))

    n = len(jobs)
    nproc = int(os.environ.get("CYCLEQA_PROCS", min(64, os.cpu_count() or 8)))
    print(f"[compute_metrics] {n} (patient, result-dir) jobs on {nproc} procs", flush=True)

    rows, t0 = [], time.time()
    with Pool(nproc) as pool:
        for i, rr in enumerate(pool.imap_unordered(one, jobs, chunksize=2), 1):
            rows.extend(rr)
            if i % 25 == 0 or i == n:
                el = time.time() - t0
                eta = el / i * (n - i)
                print(f"  [{i:4d}/{n}]  {100*i/n:5.1f}%   elapsed {el:6.1f}s   eta {eta:6.1f}s",
                      flush=True)

    df = pd.DataFrame(rows).sort_values(["dataset", "model", "patient_id"])
    out = os.path.join(ROOT, "metis", "results", "patient_metrics.csv")
    df.to_csv(out, index=False)
    print(f"\n[compute_metrics] wrote {out}  ({len(df)} rows)\n", flush=True)

    print(df.groupby(["dataset", "model"])[["dice_RV", "dice_Myo", "dice_LV", "ef_abs_err"]]
            .agg(["mean", "size"]).round(3).to_string())
    print("\n--- MedSAM2-dual on MnM, by vendor ---")
    m = df[(df.model == "MedSAM2-dual") & (df.dataset == "MnM")]
    print(m.groupby("vendor")[["dice_RV", "dice_Myo", "dice_LV", "ef_abs_err"]]
           .agg(["mean", "size"]).round(3).to_string())
    print("\n--- failure rate (any structure Dice < 0.70), MedSAM2-dual ---")
    for ds in ("ACDC", "MnM"):
        s = df[(df.model == "MedSAM2-dual") & (df.dataset == ds)]
        print(f"  {ds:5s} n={len(s):3d}  failure={100 * (s.dice_min < 0.70).mean():.1f}%")


if __name__ == "__main__":
    main()
