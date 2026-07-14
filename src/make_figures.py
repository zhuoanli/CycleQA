"""The five figures. Every number is read from results/*.csv; nothing is typed by hand.

Design notes. LNCS proceedings are frequently read in greyscale, so identity is carried by
marker shape and hatch in addition to hue -- colour never encodes identity alone. The
four-manufacturer palette was validated for lightness band, chroma floor, adjacent-pair
colour-vision-deficiency separation (>= 12 dE across deuteran/protan/tritan) and >= 3:1
contrast against the white surface.

The three-way cohort split in Fig. 1 is deliberate. The development cohort is acquired on
Siemens scanners, so a bare development-vs-external comparison would confound a change of
dataset with a change of manufacturer. Splitting the external cohort into same-manufacturer
and unseen-manufacturer separates them.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLS, ROOT, patient_slices

RES = os.path.join(ROOT, "metis", "results")
FIG = os.path.join(ROOT, "metis", "figures")
os.makedirs(FIG, exist_ok=True)

VENDORS = ["Siemens", "Philips", "GE", "Canon"]
COL = {"Siemens": "#0173B2", "Philips": "#B86B00", "GE": "#029E73", "Canon": "#CC78BC"}
MRK = {"Siemens": "o", "Philips": "s", "GE": "^", "Canon": "D"}
HATCH = {"Siemens": "", "Philips": "///", "GE": "...", "Canon": "xxx"}
DEV = "#4D4D4D"
INK, MUTED, GRID = "#1A1A1A", "#5C5C5C", "#D8D8D8"

# structure colours for the mask overlays (RV / Myo / LV)
STRUCT = {1: "#E8000B", 2: "#FFC400", 3: "#00A0E0"}
STRUCT_NAME = {1: "RV", 2: "Myocardium", 3: "LV"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), facecolor="white")
    plt.close(fig)
    print(f"  wrote figures/{name}")


def cohorts():
    m = pd.read_csv(os.path.join(RES, "patient_metrics.csv"))
    m = m[m.model == "MedSAM2-dual"]
    return m[m.dataset == "ACDC"], m[m.dataset == "MnM"]


# ---------------------------------------------------------------- Figure 1
def fig1_vendor():
    dev, ext = cohorts()
    groups = [dev, ext[ext.vendor == "Siemens"]] + [ext[ext.vendor == v] for v in
                                                    ["Philips", "GE", "Canon"]]
    labels = ["ACDC (Siemens)", "M&Ms Siemens", "M&Ms Philips", "M&Ms GE", "M&Ms Canon"]
    cols = [DEV, COL["Siemens"], COL["Philips"], COL["GE"], COL["Canon"]]
    hats = ["", "", HATCH["Philips"], HATCH["GE"], HATCH["Canon"]]
    x = np.arange(5)

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2), gridspec_kw={"wspace": 0.34})

    def panel(ax, vals, fmt, ymax, pad, ylabel, title):
        b = ax.bar(x, vals, 0.66, color=cols, hatch=hats, edgecolor="white",
                   linewidth=0.6, zorder=2)
        b[0].set_edgecolor(INK); b[0].set_linewidth(1.0)
        for xi, vi in zip(x, vals):
            ax.text(xi, vi + pad, fmt.format(vi), ha="center", fontsize=7.2, color=INK)
        ax.axvspan(1.5, 4.5, color="#F2F2F2", zorder=0)
        ax.text(3.0, ymax * 0.97, "manufacturers not seen in development", fontsize=6.2,
                color=MUTED, ha="center", va="top", style="italic")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=32, ha="right")
        ax.set_ylim(0, ymax); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9, pad=7)

    panel(axes[0], [g[["dice_RV", "dice_Myo", "dice_LV"]].mean().mean() for g in groups],
          "{:.3f}", 1.08, 0.035, "Mean Dice (RV, Myo, LV)", "(a) Segmentation accuracy")
    panel(axes[1], [g.ef_abs_err.mean() for g in groups],
          "{:.1f}", 43, 1.1, "Absolute EF error (percentage points)",
          "(b) Ejection-fraction error")
    panel(axes[2], [100 * (g.dice_min < 0.70).mean() for g in groups],
          "{:.0f}", 90, 2.2, "Studies with any Dice < 0.70  (%)", "(c) Failure rate")

    fig.suptitle("Benchmark accuracy does not transfer across scanner manufacturers: "
                 "a new hospital on the same scanner costs nothing, a new scanner costs everything",
                 fontsize=9.5, y=1.02)
    save(fig, "fig1_vendor.png")


# ---------------------------------------------------------------- Figure 2
def fig2_qualitative():
    """The method, seen directly on the images.

    Two cases at a mid-cycle frame, where neither propagation is anchored. The forward and
    backward passes are shown side by side, together with the voxels on which they disagree --
    which is the only thing CycleQA is allowed to look at.
    """
    scores = pd.read_csv(os.path.join(RES, "qc_scores.csv")).set_index("patient_id")
    curves = np.load(os.path.join(RES, "disagreement_curves.npz"))

    cases = [("I5L3S2", "Siemens", "PASS"), ("B3E2W8", "Canon", "REVIEW")]
    slices_mnm = patient_slices("MnM")

    fig, axes = plt.subplots(2, 5, figsize=(10.4, 4.4),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.85],
                                          "wspace": 0.10, "hspace": 0.10})
    for r in range(2):                       # give the A(t) panel room for its y-label
        b = axes[r, 4].get_position()
        axes[r, 4].set_position([b.x0 + 0.045, b.y0, b.width - 0.045, b.height])

    for r, (pid, vendor, verdict) in enumerate(cases):
        files = slices_mnm[pid]
        z = len(files) // 2                                   # mid-ventricular slice
        d = np.load(files[z], allow_pickle=True)
        p = np.load(os.path.join(ROOT, "results/medsam2_mnm",
                                 os.path.basename(files[z])), allow_pickle=True)
        ed, es = p["ed_pred"], p["es_pred"]
        ei, si = int(p["ed_idx"]), int(p["es_idx"])
        T = min(ed.shape[0], es.shape[0])
        lo, hi = sorted((ei, si))
        t = int((lo + hi) / 2) if hi > lo + 1 else min(T - 1, lo + 1)   # mid-cycle frame

        img = d["frames"][t, 0].astype(np.float32)
        nz = ((ed[t] > 0) | (es[t] > 0))
        ys, xs = np.where(nz)
        if ys.size:
            cy, cx = int(ys.mean()), int(xs.mean())
        else:
            cy = cx = 256
        h = 90
        sl = (slice(max(0, cy - h), cy + h), slice(max(0, cx - h), cx + h))

        def overlay(ax, mask, title=None):
            ax.imshow(img[sl], cmap="gray", interpolation="nearest")
            for c, col in STRUCT.items():
                m = (mask[sl] == c)
                if m.sum():
                    ax.contour(m, levels=[0.5], colors=[col], linewidths=1.1)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_edgecolor(MUTED); s.set_linewidth(0.6)
            if title:
                ax.set_title(title, fontsize=8, pad=4)

        overlay(axes[r, 0], np.zeros_like(ed[t]),
                "Image (mid-cycle)" if r == 0 else None)
        overlay(axes[r, 1], ed[t], "Forward pass\n(anchored at ED)" if r == 0 else None)
        overlay(axes[r, 2], es[t], "Backward pass\n(anchored at ES)" if r == 0 else None)

        # disagreement: voxels where the two passes assign different labels
        ax = axes[r, 3]
        ax.imshow(img[sl], cmap="gray", interpolation="nearest")
        dis = (ed[t] != es[t])[sl]
        ax.imshow(np.ma.masked_where(~dis, dis), cmap=ListedColormap(["#E8000B"]),
                  alpha=0.55, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_edgecolor(MUTED); s.set_linewidth(0.6)
        if r == 0:
            ax.set_title("Disagreement\n(all CycleQA sees)", fontsize=8, pad=4)

        # A(t) across the cycle
        ax = axes[r, 4]
        A = curves[f"MnM/{pid}"]                       # (3, T)
        tt = np.arange(A.shape[1])
        for ci, name in enumerate(CLS.values()):
            ax.plot(tt, A[ci], lw=1.3, color=list(STRUCT.values())[ci], label=name)
        ax.axvline(t, color=MUTED, lw=0.9, ls=":")
        ax.set_ylim(0, 1.28); ax.set_xlim(0, A.shape[1] - 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("Agreement $A(t)$", fontsize=7.5)
        ax.tick_params(labelsize=7)
        s = scores.loc[pid]
        ax.text(0.97, 0.04,
                f"CycleQA risk = {s.qc_LogReg:.2f}\ntrue Dice = {s.dice_mean:.3f}\n"
                f"EF error = {s.ef_abs_err:.1f} pp",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                color=INK, linespacing=1.5)
        if r == 0:
            ax.set_title("Agreement over the cardiac cycle", fontsize=8, pad=4)
            ax.legend(fontsize=6.8, loc="upper left", ncol=3, handlelength=1.1,
                      columnspacing=0.9, handletextpad=0.3, borderpad=0.2)
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Cardiac phase (frame)", fontsize=7.5)

        # row label
        colr = "#1A7F37" if verdict == "PASS" else "#C42B1C"
        axes[r, 0].text(-0.12, 0.5, f"{vendor}\n{verdict}", transform=axes[r, 0].transAxes,
                        rotation=90, ha="center", va="center", fontsize=8.5,
                        color=colr, weight="bold")

    fig.suptitle("The two propagations diverge exactly where the segmentation fails; "
                 "no reference mask is needed to see it", fontsize=9.5, y=0.99)
    save(fig, "fig2_qualitative.png")


# ---------------------------------------------------------------- Figure 3
def fig3_estimation():
    feat = pd.read_csv(os.path.join(RES, "patient_features.csv"))
    met = pd.read_csv(os.path.join(RES, "patient_metrics.csv"))
    met = met[met.model == "MedSAM2-dual"]
    d = feat.merge(met, on=["patient_id", "dataset"], validate="1:1")
    est = pd.read_csv(os.path.join(RES, "dice_estimation.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw={"wspace": 0.26})
    for ax, ds, title in [(axes[0], "ACDC", "(a) Development: ACDC ($n$ = 100)"),
                          (axes[1], "MnM", "(b) External: M&Ms ($n$ = 320)")]:
        g = d[d.dataset == ds]
        ax.plot([0.3, 1.0], [0.3, 1.0], color=MUTED, lw=0.9, ls="--", zorder=1)
        if ds == "ACDC":
            ax.scatter(g.dice_mean, g.anchor_agree_mean_ALL, s=20, marker="o",
                       facecolor=DEV, edgecolor="white", linewidth=0.4, alpha=0.85, zorder=2)
        else:
            for v in VENDORS:
                s = g[g.vendor == v]
                ax.scatter(s.dice_mean, s.anchor_agree_mean_ALL, s=22, marker=MRK[v],
                           facecolor=COL[v], edgecolor="white", linewidth=0.4,
                           alpha=0.85, label=v, zorder=2)
            ax.legend(loc="upper left", fontsize=7, handletextpad=0.2, borderpad=0.2,
                      labelspacing=0.3, title="Manufacturer", title_fontsize=7)
        r = est[(est.scope == ds) & (est.subgroup == "ALL")].iloc[0]
        ax.text(0.97, 0.05, f"$r$ = {r.pearson_r:.3f}\nMAE = {r.mae:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                color=INK, linespacing=1.6)
        ax.set_xlabel("True Dice   (requires a reference mask)")
        ax.set_title(title, fontsize=9, pad=7)
        ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.0); ax.set_aspect("equal")
    axes[0].set_ylabel("CycleQA estimate\n(agreement of the two propagations;\n"
                       "no reference mask, no training)")
    fig.suptitle("Segmentation quality is estimable from the predictions alone",
                 fontsize=10, y=1.01)
    save(fig, "fig3_estimation.png")


# ---------------------------------------------------------------- Figure 4
def fig4_detection():
    s = pd.read_csv(os.path.join(RES, "qc_scores.csv"))
    perf = pd.read_csv(os.path.join(RES, "qc_performance.csv"))
    ood = pd.read_csv(os.path.join(RES, "ood_control.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), gridspec_kw={"wspace": 0.42})

    # (a) ROC per manufacturer
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls="--", zorder=1)
    for v in VENDORS:
        g = s[s.vendor == v]
        if g["fail"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(g["fail"], g["qc_LogReg"])
        a = perf[(perf.model == "LogReg") & (perf.subgroup == v)].auroc.iloc[0]
        star = "" if v == "Siemens" else "*"
        ax.plot(fpr, tpr, color=COL[v], lw=1.8, marker=MRK[v], markevery=0.22,
                markersize=4, markeredgecolor="white", markeredgewidth=0.4,
                label=f"{v}{star}   {a:.3f}", zorder=2)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("(a) Failure detection by manufacturer", fontsize=9, pad=7)
    lg = ax.legend(fontsize=7, loc="lower right", handlelength=1.6, labelspacing=0.35,
                   title="AUROC        * not seen in development", title_fontsize=6.6)
    lg._legend_box.align = "left"
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    # (b) the control -- two detectors, two targets
    ax = axes[1]
    dets = ["Image-OOD (Mahalanobis)", "CycleQA (LogReg)"]
    short = ["Appearance-based OOD\n(reads the image)", "CycleQA\n(reads the mask)"]
    tg = ["is this Canon?", "is this segmentation wrong?"]
    x = np.array([0.0, 1.5])                     # widely separated groups
    w = 0.42
    for k, (t, fc, hh, lab) in enumerate(zip(
            tg, ["white", "#0173B2"], ["///", ""],
            ["Identifies an unseen manufacturer", "Identifies a failed segmentation"])):
        vals = [ood[(ood.detector == d) & (ood.target == t)].auroc.iloc[0] for d in dets]
        ax.bar(x + (k - 0.5) * w * 1.06, vals, w, facecolor=fc, hatch=hh,
               edgecolor=INK if fc == "white" else "white", linewidth=0.9,
               label=lab, zorder=2)
        for xi, vi in zip(x + (k - 0.5) * w * 1.06, vals):
            ax.text(xi, vi + 0.022, f"{vi:.2f}", ha="center", fontsize=7.4, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.4)
    ax.set_xlim(-0.62, 2.12)
    ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("(b) Error detection vs. manufacturer detection", fontsize=9, pad=7)
    ax.legend(fontsize=6.8, loc="upper center", ncol=1, handlelength=1.3,
              labelspacing=0.3, borderpad=0.2)

    # (c) calibration
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls="--", zorder=1,
            label="perfect calibration")
    pt, pp = calibration_curve(s["fail"], s["qc_LogReg"], n_bins=8, strategy="quantile")
    ax.plot(pp, pt, color="#0173B2", lw=1.8, marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=0.5, label="CycleQA", zorder=2)
    br = perf[(perf.model == "LogReg") & (perf.scope == "MnM (external)")
              & (perf.subgroup == "ALL")].brier.iloc[0]
    ax.text(0.97, 0.05, f"Brier = {br:.3f}", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8.5, color=INK)
    ax.set_xlabel("Predicted failure risk")
    ax.set_ylabel("Observed failure rate")
    ax.set_title("(c) Risk calibration", fontsize=9, pad=7)
    ax.legend(fontsize=7, loc="upper left", handlelength=1.6, labelspacing=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    fig.suptitle("Failure detection transfers to manufacturers absent from development",
                 fontsize=10, y=1.02)
    save(fig, "fig4_detection.png")


# ---------------------------------------------------------------- Figure 5
def fig5_action():
    rc = pd.read_csv(os.path.join(RES, "risk_coverage.csv"))
    rs = pd.read_csv(os.path.join(RES, "reselection.csv"))
    rc = rc[rc.model == "LogReg"].sort_values("coverage")

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), gridspec_kw={"wspace": 0.40})

    # (a) selective referral
    ax = axes[0]
    ax.plot(100 * rc.coverage, rc.residual_ef_mae, color="#0173B2", lw=2.0,
            marker="o", markersize=5.5, markeredgecolor="white", markeredgewidth=0.6, zorder=2)
    ax.set_xlabel("Studies reported automatically (%)")
    ax.set_ylabel("Residual EF error among them (pp)")
    ax.invert_xaxis(); ax.set_ylim(15, 25)
    ax.set_title("(a) Selective referral", fontsize=9, pad=7)
    y80 = rc[rc.coverage == 0.8].residual_ef_mae.iloc[0]
    ax.annotate("riskiest 20% sent\nfor human review", xy=(80, y80), xytext=(66, 23.4),
                fontsize=7, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.9, color=MUTED))

    # (b) reselection
    ax = axes[1]
    order = ["always-bidir (current system)", "always-ES (best fixed, hindsight)",
             "CycleQA-guided reselection", "oracle: min EF error"]
    lbl = ["Current\nsystem", "Best fixed\n(needs labels)", "CycleQA\nreselection",
           "Oracle\n(upper bound)"]
    g = rs[rs.scope == "ALL"].set_index("strategy")
    vals = [g.loc[o, "ef_mae"] for o in order]
    bars = ax.bar(range(4), vals, 0.64, color=["white", "#BFBFBF", "#0173B2", "white"],
                  hatch=["", "", "", "///"], edgecolor=INK, linewidth=0.9, zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.7, f"{v:.1f}", ha="center", fontsize=7.6, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(lbl, fontsize=7.2)
    ax.set_ylabel("EF error (pp)"); ax.set_ylim(0, 28)
    ax.set_title("(b) QC-guided reselection", fontsize=9, pad=7)

    # (c) per manufacturer
    ax = axes[2]
    x = np.arange(4); w = 0.38
    cur = [rs[(rs.scope == v) & (rs.strategy == order[0])].ef_mae.iloc[0] for v in VENDORS]
    new = [rs[(rs.scope == v) & (rs.strategy == order[2])].ef_mae.iloc[0] for v in VENDORS]
    ax.bar(x - w / 2, cur, w * 0.92, facecolor="white", edgecolor=INK, linewidth=0.9,
           hatch="///", label="current system", zorder=2)
    ax.bar(x + w / 2, new, w * 0.92, facecolor="#0173B2", edgecolor="white",
           linewidth=0.5, label="with CycleQA reselection", zorder=2)
    for xi, vi in zip(x + w / 2, new):
        ax.text(xi, vi + 1.0, f"{vi:.1f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(VENDORS, fontsize=7.6)
    ax.set_ylabel("EF error (pp)"); ax.set_ylim(0, 43)
    ax.set_title("(c) Reselection gain by manufacturer", fontsize=9, pad=7)
    ax.legend(fontsize=7, loc="upper left", handlelength=1.4, labelspacing=0.3)

    fig.suptitle("Acting on the risk score requires no labels, no retraining "
                 "and no additional inference", fontsize=10, y=1.02)
    save(fig, "fig5_action.png")


if __name__ == "__main__":
    fig1_vendor()
    fig2_qualitative()
    fig3_estimation()
    fig4_detection()
    fig5_action()
    print("\n[make_figures] done")
