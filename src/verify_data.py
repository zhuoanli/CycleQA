"""Gate 0a -- blocking data verification for CycleQA.

Canon's myocardial Dice of 0.436 is low enough that it could be an ED/ES indexing
bug rather than domain shift. If it is, the cross-vendor narrative is false. This
script decides GO / NO-GO before any manuscript claim is made.

Checks, stratified by vendor:
  1. ed_idx / es_idx in preprocessed_mnm NPZs vs the ED / ES columns of the M&Ms
     metadata CSV.
  2. GT ed_mask has a larger LV cavity than GT es_mask (i.e. the two masks are not
     swapped).
  3. Label convention is 1=RV, 2=Myo, 3=LV in BOTH datasets, established
     geometrically: the LV is the pool fully enclosed by the myocardial ring.
  4. Whether the bidirectional-disagreement signal survives when restricted to
     mid-cycle frames -- the purest test, since neither propagation is anchored there.

Run:  python metis/src/verify_data.py
"""
import collections
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MNM_META = os.path.join(ROOT, "MnM", "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv")
CLS = {1: "RV", 2: "Myo", 3: "LV"}

failures = []


def patient_slices(prep_dir):
    d = collections.defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, prep_dir, "*.npz")):
        d[os.path.basename(f).split("_slice")[0]].append(f)
    return {k: sorted(v) for k, v in d.items()}


def dice(a, b, c):
    p, g = (a == c), (b == c)
    if p.sum() == 0 and g.sum() == 0:
        return np.nan
    return 2 * (p & g).sum() / (p.sum() + g.sum() + 1e-8)


# ---------------------------------------------------------------- check 3
def check_label_convention():
    """LV is the class whose boundary neighbourhood is dominated by myocardium."""
    print("\n" + "=" * 78)
    print("CHECK 3 -- label convention (geometric, not documentation-based)")
    print("=" * 78)
    for name, prep in [("ACDC", "preprocessed"), ("M&M", "preprocessed_mnm")]:
        enc = {1: [], 3: []}
        for f in sorted(glob.glob(os.path.join(ROOT, prep, "*.npz")))[:400:7]:
            gt = np.load(f, allow_pickle=True)["ed_mask"]
            if not {1, 2, 3} <= set(np.unique(gt)):
                continue
            for lab in (1, 3):
                m = gt == lab
                if m.sum() < 20:
                    continue
                ring = binary_dilation(m, iterations=2) & ~m
                if ring.sum():
                    enc[lab].append(((gt == 2) & ring).sum() / ring.sum())
        e1, e3 = np.nanmean(enc[1]), np.nanmean(enc[3])
        lv = 3 if e3 > e1 else 1
        ok = lv == 3
        print(f"  {name:5s}  label1 myo-enclosure={e1:.3f}  label3 myo-enclosure={e3:.3f}"
              f"  -> LV = label{lv}   {'OK (1=RV,2=Myo,3=LV)' if ok else '*** MISMATCH ***'}")
        if not ok:
            failures.append(f"{name}: label convention is NOT 1=RV,2=Myo,3=LV")


# ---------------------------------------------------------------- checks 1 & 2
def check_ed_es_indices():
    print("\n" + "=" * 78)
    print("CHECK 1+2 -- M&M ED/ES frame indices vs metadata, and ED/ES mask ordering")
    print("=" * 78)
    meta = pd.read_csv(MNM_META).set_index("External code")
    pats = patient_slices("preprocessed_mnm")
    rows = []
    for pid, files in pats.items():
        if pid not in meta.index:
            continue
        d = np.load(files[0], allow_pickle=True)
        # LV cavity volume from GT, summed over all slices of this patient
        lv_ed = lv_es = 0
        for f in files:
            dd = np.load(f, allow_pickle=True)
            lv_ed += int((dd["ed_mask"] == 3).sum())
            lv_es += int((dd["es_mask"] == 3).sum())
        rows.append({
            "pid": pid,
            "vendor": meta.loc[pid, "VendorName"],
            "npz_ed": int(d["ed_idx"]), "npz_es": int(d["es_idx"]),
            "meta_ed": int(meta.loc[pid, "ED"]), "meta_es": int(meta.loc[pid, "ES"]),
            "lv_ed_vox": lv_ed, "lv_es_vox": lv_es,
        })
    df = pd.DataFrame(rows)
    df["idx_match"] = (df.npz_ed == df.meta_ed) & (df.npz_es == df.meta_es)
    df["edv_gt_esv"] = df.lv_ed_vox > df.lv_es_vox

    print(f"\n  n patients matched to metadata: {len(df)}\n")
    g = df.groupby("vendor").agg(
        n=("pid", "size"),
        idx_match_pct=("idx_match", lambda s: 100 * s.mean()),
        edv_gt_esv_pct=("edv_gt_esv", lambda s: 100 * s.mean()),
    ).round(1)
    print(g.to_string())

    for vendor, r in g.iterrows():
        if r.idx_match_pct < 99:
            failures.append(f"{vendor}: only {r.idx_match_pct:.0f}% of ED/ES indices match metadata")
        if r.edv_gt_esv_pct < 95:
            failures.append(f"{vendor}: only {r.edv_gt_esv_pct:.0f}% of patients have EDV > ESV "
                            f"(ED/ES masks may be swapped)")
    df.to_csv(os.path.join(ROOT, "metis", "results", "gate0a_ed_es_check.csv"), index=False)
    return df


# ---------------------------------------------------------------- check 4
def check_disagreement_signal():
    """Does bidirectional disagreement predict true error -- including mid-cycle only?"""
    print("\n" + "=" * 78)
    print("CHECK 4 -- bidirectional-propagation disagreement vs true error")
    print("        (ALL frames, and MID-CYCLE only: the purest test -- neither pass anchored)")
    print("=" * 78)
    for tag, prep, pred in [("ACDC", "preprocessed", "results/medsam2"),
                            ("M&M ", "preprocessed_mnm", "results/medsam2_mnm")]:
        pats = patient_slices(prep)
        ids = sorted(pats)[::max(1, len(pats) // 50)][:50]
        rows = []
        for pid in ids:
            agree_all, agree_mid, true = [], [], []
            for f in pats[pid]:
                pf = os.path.join(ROOT, pred, os.path.basename(f))
                if not os.path.exists(pf):
                    continue
                d = np.load(f, allow_pickle=True)
                r = np.load(pf, allow_pickle=True)
                ed, es, bi = r["ed_pred"], r["es_pred"], r["bidir"]
                ei, si = int(r["ed_idx"]), int(r["es_idx"])
                T = len(bi)
                lo, hi = sorted((ei, si))
                # mid-cycle = frames at least 25% of the anchor separation from both anchors
                pad = max(1, int(0.25 * max(1, hi - lo)))
                mid = [t for t in range(T) if lo + pad <= t <= hi - pad]
                for c in CLS:
                    a = [dice(ed[t], es[t], c) for t in range(T)]
                    agree_all += a
                    agree_mid += [a[t] for t in mid]
                    for idx, gt in ((ei, d["ed_mask"]), (si, d["es_mask"])):
                        if idx < T:
                            true.append(dice(bi[idx], gt, c))
            if true:
                rows.append({
                    "agree_all": np.nanmean(agree_all) if agree_all else np.nan,
                    "agree_mid": np.nanmean(agree_mid) if agree_mid else np.nan,
                    "true": np.nanmean(true),
                })
        df = pd.DataFrame(rows).dropna()
        print(f"\n  {tag}  (n={len(df)} patients)")
        for col, label in [("agree_all", "ALL frames"), ("agree_mid", "MID-CYCLE only")]:
            r, p = pearsonr(df[col], df["true"])
            rs, _ = spearmanr(df[col], df["true"])
            mae = np.abs(df[col] - df["true"]).mean()
            print(f"    {label:16s}  Pearson r={r:+.3f} (p={p:.1e})  Spearman={rs:+.3f}  "
                  f"| disagreement-as-Dice MAE={mae:.3f}")


# ---------------------------------------------------------------- check 5
def check_unet_checkpoint():
    print("\n" + "=" * 78)
    print("CHECK 5 -- which checkpoint produced results/unet_mnm/ ?")
    print("        (unet/best_model.pth = ACDC-only, CLEAN zero-shot transfer)")
    print("        (unet_combined/best_model.pth = ACDC+M&Ms-2, belongs in the 3c reference row)")
    print("=" * 78)
    import subprocess
    for p in ["results/unet/best_model.pth", "results/unet_combined/best_model.pth",
              "results/unet_mnm", "results/dinov2_mnm"]:
        full = os.path.join(ROOT, p)
        if os.path.exists(full):
            mt = subprocess.run(["stat", "-c", "%y", full], capture_output=True, text=True).stdout.strip()
            print(f"  {p:40s}  mtime {mt[:19]}")
        else:
            print(f"  {p:40s}  MISSING")
    print("\n  -> unet_mnm predictions must POST-DATE the checkpoint they came from.")
    print("     Resolve by reading jobs/job_supervised_mnm.sh (which --ckpt did it pass?).")


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "metis", "results"), exist_ok=True)
    check_label_convention()
    check_ed_es_indices()
    check_disagreement_signal()
    check_unet_checkpoint()

    print("\n" + "=" * 78)
    if failures:
        print("*** GATE 0a: NO-GO ***")
        for f in failures:
            print("  - " + f)
        print("\nDo NOT write any manuscript claim until these are resolved.")
        sys.exit(1)
    print("*** GATE 0a: GO -- vendor gradient is not an indexing artifact ***")
    print("=" * 78)
