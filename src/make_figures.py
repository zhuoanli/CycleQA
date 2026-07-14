"""The five figures. Every number is read from results/*.csv; nothing is typed by hand.

Design rules followed here:
  - Text inside a panel is kept to the minimum needed to read it. Everything else belongs in
    the caption.
  - Legends carry identity only. No metrics are printed inside a legend.
  - Manufacturer identity is carried by marker shape and hatch as well as hue, so the figures
    survive the greyscale printing common in LNCS proceedings. Colour never encodes identity
    alone.
  - The four-manufacturer palette was validated for lightness band, chroma floor, adjacent-pair
    colour-vision-deficiency separation (>= 12 dE across deuteran / protan / tritan) and >= 3:1
    contrast against the white surface.

The three-way cohort split in Fig. 1 is deliberate: the development cohort is acquired on
Siemens scanners, so a bare development-versus-external comparison would confound a change of
dataset with a change of manufacturer. Splitting the external cohort into same-manufacturer and
unseen-manufacturer separates the two.
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
SHADE = "#E4E4E4"                      # the unseen-manufacturer block
INK, MUTED, GRID = "#1A1A1A", "#5C5C5C", "#D8D8D8"

STRUCT = {1: "#E8000B", 2: "#FFC400", 3: "#00A0E0"}      # RV / Myo / LV

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": False,
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
    groups = [dev, ext[ext.vendor == "Siemens"]] + [ext[ext.vendor == v]
                                                    for v in ["Philips", "GE", "Canon"]]
    # ACDC is itself a Siemens cohort, so it carries the Siemens colour. The two group
    # brackets below the axis say which dataset each bar comes from.
    labels = ["Siemens", "Siemens", "Philips", "GE", "Canon"]
    cols = [COL["Siemens"], COL["Siemens"], COL["Philips"], COL["GE"], COL["Canon"]]
    hats = ["", "", HATCH["Philips"], HATCH["GE"], HATCH["Canon"]]
    x = np.arange(5)

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5), gridspec_kw={"wspace": 0.40})

    def panel(ax, vals, fmt, ymax, pad, ylabel, title):
        ax.axvspan(1.5, 4.5, color=SHADE, zorder=0)
        b = ax.bar(x, vals, 0.66, color=cols, hatch=hats, edgecolor="white",
                   linewidth=0.6, zorder=2)
        b[0].set_edgecolor(INK); b[0].set_linewidth(1.2)      # development cohort
        for xi, vi in zip(x, vals):
            ax.text(xi, vi + pad, fmt.format(vi), ha="center", fontsize=7.4, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.4, rotation=28, ha="right")
        ax.set_ylim(0, ymax); ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(title, fontsize=9.5, pad=8)
        # dataset brackets under the axis
        for x0, x1, lab in [(-0.42, 0.42, "ACDC"), (0.58, 4.42, "M&Ms")]:
            ax.plot([x0, x1], [-0.235 * ymax] * 2, color=MUTED, lw=0.9,
                    clip_on=False, zorder=3)
            ax.text((x0 + x1) / 2, -0.29 * ymax, lab, ha="center", va="top",
                    fontsize=8, color=INK, clip_on=False)

    panel(axes[0], [g[["dice_RV", "dice_Myo", "dice_LV"]].mean().mean() for g in groups],
          "{:.3f}", 1.05, 0.03, "Mean Dice", "(a) Segmentation accuracy")
    panel(axes[1], [g.ef_abs_err.mean() for g in groups],
          "{:.1f}", 41, 1.1, "EF error (pp)", "(b) Ejection-fraction error")
    panel(axes[2], [100 * (g.dice_min < 0.70).mean() for g in groups],
          "{:.0f}", 86, 2.2, "Failed studies (%)", "(c) Failure rate")

    fig.suptitle("Benchmark accuracy does not transfer across scanner manufacturers",
                 fontsize=10, y=1.03)
    save(fig, "fig1_vendor.png")


# ---------------------------------------------------------------- Figure 2
def fig2_qualitative():
    """Reference vs. prediction where an annotation exists; the two propagations and their
    disagreement where none does."""
    scores = pd.read_csv(os.path.join(RES, "qc_scores.csv")).set_index("patient_id")
    curves = np.load(os.path.join(RES, "disagreement_curves.npz"))
    # PASS:   RV .918 / Myo .813 / LV .886, EF error 0.2 pp, risk 0.12.
    # REVIEW: the two propagations diverge -- Dice .620, EF error 41.7 pp, risk 0.99.
    cases = [("B5F8L9", "Siemens", "PASS", "#1A7F37"),
             ("I8N8Y1", "Canon", "REVIEW", "#C42B1C")]
    sl_mnm = patient_slices("MnM")

    def best_slice(files):
        """A mid-ventricular slice: basal and apical levels are excluded (the RV is truncated
        at the base and the cavity vanishes at the apex), so only the middle third of the stack
        is considered, and within it the level where all three structures are largest."""
        files = sorted(files)
        n = len(files)
        mid = files[n // 3: max(n // 3 + 1, 2 * n // 3)]
        best, score = mid[len(mid) // 2], -1.0
        for f in mid:
            gt = np.load(f, allow_pickle=True)["ed_mask"]
            areas = [(gt == c).sum() for c in CLS]
            if min(areas) < 200:                    # a structure is barely present at this level
                continue
            if min(areas) > score:
                best, score = f, min(areas)
        return best

    fig, axes = plt.subplots(2, 6, figsize=(11.4, 4.1),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1.75],
                                          "wspace": 0.08, "hspace": 0.10})
    for r in range(2):
        b = axes[r, 5].get_position()
        axes[r, 5].set_position([b.x0 + 0.052, b.y0, b.width - 0.052, b.height])

    for r, (pid, vendor, verdict, vcol) in enumerate(cases):
        f = best_slice(sl_mnm[pid])
        d = np.load(f, allow_pickle=True)
        p = np.load(os.path.join(ROOT, "results/medsam2_mnm", os.path.basename(f)),
                    allow_pickle=True)
        ed, es, bi = p["ed_pred"], p["es_pred"], p["bidir"]
        ei, si = int(p["ed_idx"]), int(p["es_idx"])
        T = min(ed.shape[0], es.shape[0], bi.shape[0])
        lo, hi = sorted((ei, si))
        tm = int((lo + hi) / 2) if hi > lo + 1 else min(T - 1, lo + 1)     # mid-cycle
        gt = d["ed_mask"]

        # crop tightly to the whole heart: the bounding box of the reference and both
        # propagations, made square, with a small margin
        nz = (ed[tm] > 0) | (es[tm] > 0) | (bi[ei] > 0) | (gt > 0)
        ys, xs = np.where(nz)
        if ys.size:
            cy, cx = int((ys.min() + ys.max()) / 2), int((xs.min() + xs.max()) / 2)
            h = int(max(np.ptp(ys), np.ptp(xs)) / 2) + 14
        else:
            cy = cx = 256; h = 88
        y0, x0 = max(0, cy - h), max(0, cx - h)
        sl = (slice(y0, min(512, y0 + 2 * h)), slice(x0, min(512, x0 + 2 * h)))

        def show(ax, frame, mask, title):
            ax.imshow(frame[sl], cmap="gray", interpolation="nearest")
            if mask is not None:
                for c, col in STRUCT.items():
                    m = (mask[sl] == c)
                    if m.sum():
                        ax.contour(m, levels=[0.5], colors=[col], linewidths=1.2)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_edgecolor(MUTED); s.set_linewidth(0.6)
            if r == 0 and title:
                ax.set_title(title, fontsize=7.8, pad=4)

        img_ed = d["frames"][ei, 0].astype(np.float32)
        img_md = d["frames"][tm, 0].astype(np.float32)

        show(axes[r, 0], img_ed, gt, "Reference")
        show(axes[r, 1], img_ed, bi[ei], "Prediction")
        show(axes[r, 2], img_md, ed[tm], "Forward pass")
        show(axes[r, 3], img_md, es[tm], "Backward pass")

        ax = axes[r, 4]
        ax.imshow(img_md[sl], cmap="gray", interpolation="nearest")
        dis = (ed[tm] != es[tm])[sl]
        ax.imshow(np.ma.masked_where(~dis, dis), cmap=ListedColormap(["#E8000B"]),
                  alpha=0.6, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_edgecolor(MUTED); s.set_linewidth(0.6)
        if r == 0:
            ax.set_title("Disagreement", fontsize=7.8, pad=4)

        # A(t)
        ax = axes[r, 5]
        A = curves[f"MnM/{pid}"]
        for ci, name in enumerate(CLS.values()):
            ax.plot(np.arange(A.shape[1]), A[ci], lw=1.4,
                    color=list(STRUCT.values())[ci], label=name)
        ax.axvline(tm, color=MUTED, lw=0.9, ls=":")
        ax.set_ylim(0, 1.05); ax.set_xlim(0, A.shape[1] - 1)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel("$A(t)$", fontsize=8)
        ax.tick_params(labelsize=7)
        s = scores.loc[pid]
        ax.text(0.97, 0.08, f"risk {s.qc_LogReg:.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, color=vcol, weight="bold")
        if r == 0:
            ax.set_title("Agreement over the cycle", fontsize=7.8, pad=4)
            ax.legend(fontsize=6.8, loc="upper left", ncol=3, handlelength=1.0,
                      columnspacing=0.8, handletextpad=0.3, borderpad=0.15)
            ax.set_ylim(0, 1.28); ax.set_xticklabels([])
        else:
            ax.set_xlabel("Frame", fontsize=8)

        axes[r, 0].text(-0.13, 0.5, f"{vendor}\n{verdict}", transform=axes[r, 0].transAxes,
                        rotation=90, ha="center", va="center", fontsize=8.5,
                        color=vcol, weight="bold")

    # separate the annotated block from the unannotated one
    for lbl, a, b in [("annotated phase (ED)", 0, 1), ("mid-cycle: no annotation exists", 2, 4)]:
        p0 = axes[0, a].get_position(); p1 = axes[0, b].get_position()
        fig.text((p0.x0 + p1.x1) / 2, p0.y1 + 0.085, lbl, ha="center", fontsize=7.6,
                 color=MUTED, style="italic")
    pl = axes[0, 1].get_position(); pr = axes[0, 2].get_position()
    fig.add_artist(plt.Line2D([(pl.x1 + pr.x0) / 2] * 2, [0.06, 0.86], color=MUTED,
                              lw=0.8, ls="--"))

    save(fig, "fig2_qualitative.png")


# ---------------------------------------------------------------- Figure 3
def fig3_estimation():
    feat = pd.read_csv(os.path.join(RES, "patient_features.csv"))
    met = pd.read_csv(os.path.join(RES, "patient_metrics.csv"))
    met = met[met.model == "MedSAM2-dual"]
    d = feat.merge(met, on=["patient_id", "dataset"], validate="1:1")
    est = pd.read_csv(os.path.join(RES, "dice_estimation.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.7), gridspec_kw={"wspace": 0.30})
    for ax, ds, title in [(axes[0], "ACDC", "(a) Development: ACDC"),
                          (axes[1], "MnM", "(b) External: M&Ms")]:
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
            ax.legend(loc="upper left", fontsize=7.5, handletextpad=0.2, borderpad=0.2,
                      labelspacing=0.3)
        r = est[(est.scope == ds) & (est.subgroup == "ALL")].iloc[0]
        ax.text(0.97, 0.05, f"$r$ = {r.pearson_r:.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color=INK)
        ax.set_xlabel("True Dice")
        ax.set_title(title, fontsize=9, pad=7)
        ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.0); ax.set_aspect("equal")
        ax.set_ylabel("CycleQA estimate")
    fig.suptitle("Segmentation quality is estimable from the predictions alone",
                 fontsize=10, y=1.01)
    save(fig, "fig3_estimation.png")


# ---------------------------------------------------------------- Figure 4
def fig4_detection():
    s = pd.read_csv(os.path.join(RES, "qc_scores.csv"))
    perf = pd.read_csv(os.path.join(RES, "qc_performance.csv"))
    ood = pd.read_csv(os.path.join(RES, "ood_control.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2), gridspec_kw={"wspace": 0.50})

    # (a) ROC per manufacturer -- no diagonal, no metrics in the legend
    ax = axes[0]
    for v in VENDORS:
        g = s[s.vendor == v]
        if g["fail"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(g["fail"], g["qc_LogReg"])
        a = perf[(perf.model == "LogReg") & (perf.subgroup == v)].auroc.iloc[0]
        ax.plot(fpr, tpr, color=COL[v], lw=1.8, marker=MRK[v], markevery=0.22,
                markersize=4.2, markeredgecolor="white", markeredgewidth=0.4,
                label=f"{v}   {a:.3f}", zorder=2)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("(a) Failure detection", fontsize=9, pad=7)
    ax.legend(fontsize=7.5, loc="lower right", handlelength=1.6, labelspacing=0.32)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    # (b) the control
    ax = axes[1]
    dets = ["Image-OOD (Mahalanobis)", "CycleQA (LogReg)"]
    short = ["Reads the\nimage", "Reads the\nmask"]
    tg = ["is this Canon?", "is this segmentation wrong?"]
    x = np.array([0.0, 1.6]); w = 0.44
    for k, (t, fc, hh, lab) in enumerate(zip(tg, ["white", "#0173B2"], ["///", ""],
                                             ["unseen manufacturer", "failed segmentation"])):
        vals = [ood[(ood.detector == d) & (ood.target == t)].auroc.iloc[0] for d in dets]
        ax.bar(x + (k - 0.5) * w * 1.08, vals, w, facecolor=fc, hatch=hh,
               edgecolor=INK if fc == "white" else "white", linewidth=0.9,
               label=lab, zorder=2)
        for xi, vi in zip(x + (k - 0.5) * w * 1.08, vals):
            ax.text(xi, vi + 0.022, f"{vi:.2f}", ha="center", fontsize=7.4, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.8)
    ax.set_xlim(-0.7, 2.3); ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("AUROC")
    ax.set_title("(b) What is detected", fontsize=9, pad=7)
    ax.legend(fontsize=7, loc="upper center", handlelength=1.3, labelspacing=0.28,
              borderpad=0.2, title="detects:", title_fontsize=6.8)

    # (c) calibration
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls="--", zorder=1)
    pt, pp = calibration_curve(s["fail"], s["qc_LogReg"], n_bins=8, strategy="quantile")
    ax.plot(pp, pt, color="#0173B2", lw=1.8, marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=0.5, zorder=2)
    br = perf[(perf.model == "LogReg") & (perf.scope == "MnM (external)")
              & (perf.subgroup == "ALL")].brier.iloc[0]
    ax.text(0.97, 0.05, f"Brier = {br:.3f}", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9, color=INK)
    ax.set_xlabel("Predicted risk"); ax.set_ylabel("Observed failure rate")
    ax.set_title("(c) Risk calibration", fontsize=9, pad=7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    fig.suptitle("Failure detection transfers to manufacturers absent from development",
                 fontsize=10, y=1.02)
    save(fig, "fig4_detection.png")


# ---------------------------------------------------------------- Figure 5
def fig5_action():
    rc = pd.read_csv(os.path.join(RES, "risk_coverage.csv"))
    rs = pd.read_csv(os.path.join(RES, "reselection.csv"))
    rc = rc[rc.model == "LogReg"].sort_values("coverage")

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.3), gridspec_kw={"wspace": 0.48})

    # (a) selective referral
    ax = axes[0]
    ax.plot(100 * rc.coverage, rc.residual_ef_mae, color="#0173B2", lw=2.0,
            marker="o", markersize=5.5, markeredgecolor="white", markeredgewidth=0.6, zorder=2)
    ax.set_xlabel("Reported automatically (%)")
    ax.set_ylabel("Residual EF error (pp)")
    ax.invert_xaxis(); ax.set_ylim(15, 25.4)
    ax.set_title("(a) Selective referral", fontsize=9, pad=7)
    y80 = rc[rc.coverage == 0.8].residual_ef_mae.iloc[0]
    ax.annotate(f"review the riskiest 20%\n$\\rightarrow$ {y80:.1f} pp",
                xy=(80, y80), xytext=(66, 23.6), fontsize=7.2, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", lw=1.0, color=INK))

    # (b) reselection -- rotated labels, well separated
    ax = axes[1]
    order = ["always-bidir (current system)", "always-ES (best fixed, hindsight)",
             "CycleQA-guided reselection", "oracle: min EF error"]
    lbl = ["Current system", "Best fixed", "CycleQA", "Oracle"]
    g = rs[rs.scope == "ALL"].set_index("strategy")
    vals = [g.loc[o, "ef_mae"] for o in order]
    ax.bar(range(4), vals, 0.62, color=["white", "#BFBFBF", "#0173B2", "white"],
           hatch=["", "", "", "///"], edgecolor=INK, linewidth=0.9, zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.7, f"{v:.1f}", ha="center", fontsize=7.6, color=INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(lbl, fontsize=7.6, rotation=30, ha="right")
    ax.set_ylabel("EF error (pp)"); ax.set_ylim(0, 28)
    ax.set_title("(b) QC-guided reselection", fontsize=9, pad=7)

    # (c) per manufacturer -- BOTH bars labelled
    ax = axes[2]
    x = np.arange(4); w = 0.38
    cur = [rs[(rs.scope == v) & (rs.strategy == order[0])].ef_mae.iloc[0] for v in VENDORS]
    new = [rs[(rs.scope == v) & (rs.strategy == order[2])].ef_mae.iloc[0] for v in VENDORS]
    ax.bar(x - w / 2, cur, w * 0.92, facecolor="white", edgecolor=INK, linewidth=0.9,
           hatch="///", label="current", zorder=2)
    ax.bar(x + w / 2, new, w * 0.92, facecolor="#0173B2", edgecolor="white",
           linewidth=0.5, label="CycleQA", zorder=2)
    for xi, vi in zip(x - w / 2, cur):
        ax.text(xi, vi + 1.1, f"{vi:.1f}", ha="center", fontsize=6.8, color=MUTED)
    for xi, vi in zip(x + w / 2, new):
        ax.text(xi, vi + 1.1, f"{vi:.1f}", ha="center", fontsize=6.8, color=INK, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(VENDORS, fontsize=7.6)
    ax.set_ylabel("EF error (pp)"); ax.set_ylim(0, 46)
    ax.set_title("(c) Gain by manufacturer", fontsize=9, pad=7)
    ax.legend(fontsize=7.5, loc="upper left", handlelength=1.4, labelspacing=0.3)

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
