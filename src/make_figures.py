"""Phase 3 -- the four figures.

Every number is read from metis/results/*.csv. Nothing is typed by hand.

Design: LNCS proceedings are often read in greyscale, so vendor identity is carried by
marker shape and hatch as well as hue -- colour never encodes identity alone. The
four-vendor palette passed all six checks of the colour formula (lightness band, chroma
floor, adjacent-pair CVD separation >= 12 dE across deuteran/protan/tritan, and >= 3:1
contrast on the white surface).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT

RES = os.path.join(ROOT, "metis", "results")
FIG = os.path.join(ROOT, "metis", "figures")
os.makedirs(FIG, exist_ok=True)

# validated categorical palette, fixed order -- never cycled
VENDORS = ["Siemens", "Philips", "GE", "Canon"]
COL = {"Siemens": "#0173B2", "Philips": "#B86B00", "GE": "#029E73", "Canon": "#CC78BC"}
MRK = {"Siemens": "o", "Philips": "s", "GE": "^", "Canon": "D"}   # greyscale fallback
HATCH = {"Siemens": "", "Philips": "///", "GE": "...", "Canon": "xxx"}
INK, MUTED, GRID = "#1A1A1A", "#5C5C5C", "#D8D8D8"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    print(f"  wrote {p}")


# ---------------------------------------------------------------- Fig 2
def fig2_dice_estimation():
    """Predicted (no ground truth) vs true Dice. The headline panel."""
    feat = pd.read_csv(os.path.join(RES, "patient_features.csv"))
    met = pd.read_csv(os.path.join(RES, "patient_metrics.csv"))
    met = met[met.model == "MedSAM2-dual"]
    d = feat.merge(met, on=["patient_id", "dataset"], validate="1:1")
    est = pd.read_csv(os.path.join(RES, "dice_estimation.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1))
    for ax, ds, title in [(axes[0], "ACDC", "ACDC (development, Siemens only)"),
                          (axes[1], "MnM", "M&Ms (external, 4 vendors)")]:
        g = d[d.dataset == ds]
        ax.plot([0.3, 1.0], [0.3, 1.0], color=MUTED, lw=0.8, ls="--", zorder=1)
        if ds == "ACDC":
            ax.scatter(g.dice_mean, g.anchor_agree_mean_ALL, s=16, marker="o",
                       facecolor=COL["Siemens"], edgecolor="white", linewidth=0.4,
                       alpha=0.85, zorder=2)
        else:
            for v in VENDORS:
                s = g[g.vendor == v]
                ax.scatter(s.dice_mean, s.anchor_agree_mean_ALL, s=18, marker=MRK[v],
                           facecolor=COL[v], edgecolor="white", linewidth=0.4,
                           alpha=0.85, label=v, zorder=2)
            ax.legend(loc="upper left", fontsize=6.5, handletextpad=0.2,
                      borderpad=0.2, labelspacing=0.25)
        r = est[(est.scope == ds) & (est.subgroup == "ALL")].iloc[0]
        ax.text(0.97, 0.06, f"$r$ = {r.pearson_r:.3f}\nMAE = {r.mae:.3f}\n$n$ = {int(r.n)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                color=INK, linespacing=1.5)
        ax.set_xlabel("True Dice  (requires ground truth)")
        ax.set_title(title, fontsize=8, pad=6)
        ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.0)
        ax.set_aspect("equal")
    axes[0].set_ylabel("CycleQA estimate\n(bidirectional agreement, no ground truth)")
    fig.suptitle("Segmentation quality is estimable without ground truth",
                 fontsize=9, y=1.02)
    save(fig, "fig2_dice_estimation.png")


# ---------------------------------------------------------------- Fig 3
def fig3_cross_vendor():
    """Benchmark accuracy does not transfer -- and the failure is vendor-aligned."""
    m = pd.read_csv(os.path.join(RES, "patient_metrics.csv"))
    m = m[m.model == "MedSAM2-dual"]
    acdc = m[m.dataset == "ACDC"]
    ext = m[m.dataset == "MnM"]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8),
                             gridspec_kw={"width_ratios": [1.3, 1, 1], "wspace": 0.42})

    # (a) Dice by structure x vendor
    ax = axes[0]
    structs = ["RV", "Myo", "LV"]
    x = np.arange(len(structs))
    w = 0.17
    handles = []
    b = ax.bar(x - 2 * w, [acdc[f"dice_{s}"].mean() for s in structs], w * 0.88,
               facecolor="white", edgecolor=INK, linewidth=0.8, label="ACDC", zorder=2)
    handles.append(b)
    for i, v in enumerate(VENDORS):
        g = ext[ext.vendor == v]
        b = ax.bar(x + (i - 1) * w, [g[f"dice_{s}"].mean() for s in structs], w * 0.88,
                   facecolor=COL[v], edgecolor="white", linewidth=0.5,
                   hatch=HATCH[v], label=v, zorder=2)
        handles.append(b)
    ax.set_xticks(x); ax.set_xticklabels(structs)
    ax.set_ylabel("Dice"); ax.set_ylim(0, 1.05)
    ax.set_title("(a) Segmentation accuracy", fontsize=8, pad=5)

    # (b) EF error -- the clinically used quantity
    ax = axes[1]
    vals = [acdc.ef_abs_err.mean()] + [ext[ext.vendor == v].ef_abs_err.mean() for v in VENDORS]
    cols = ["white"] + [COL[v] for v in VENDORS]
    hats = [""] + [HATCH[v] for v in VENDORS]
    edge = [INK] + ["white"] * 4
    bars = ax.bar(range(5), vals, 0.68, color=cols, hatch=hats, zorder=2)
    for b, e in zip(bars, edge):
        b.set_edgecolor(e); b.set_linewidth(0.8 if e == INK else 0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.9, f"{v:.1f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(5)); ax.set_xticklabels(["ACDC"] + VENDORS, rotation=40, ha="right",
                                                fontsize=6.5)
    ax.set_ylabel("EF error (pp)")
    ax.set_ylim(0, 42)
    ax.set_title("(b) Clinical biomarker error", fontsize=8, pad=5)

    # (c) failure rate
    ax = axes[2]
    fr = [100 * (acdc.dice_min < 0.70).mean()] + \
         [100 * (ext[ext.vendor == v].dice_min < 0.70).mean() for v in VENDORS]
    bars = ax.bar(range(5), fr, 0.68, color=cols, hatch=hats, zorder=2)
    for b, e in zip(bars, edge):
        b.set_edgecolor(e); b.set_linewidth(0.8 if e == INK else 0.5)
    for i, v in enumerate(fr):
        ax.text(i, v + 1.8, f"{v:.0f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(5)); ax.set_xticklabels(["ACDC"] + VENDORS, rotation=40, ha="right",
                                                fontsize=6.5)
    ax.set_ylabel("Any Dice < 0.70  (%)")
    ax.set_ylim(0, 88)
    ax.set_title("(c) Failure rate", fontsize=8, pad=5)

    fig.legend(handles=handles, labels=["ACDC"] + VENDORS, ncol=5, fontsize=6.8,
               loc="lower center", bbox_to_anchor=(0.5, -0.13), handlelength=1.3,
               handletextpad=0.35, columnspacing=1.3)
    fig.suptitle("Dice degrades modestly off-vendor; the clinical biomarker collapses",
                 fontsize=9, y=1.03)
    save(fig, "fig3_cross_vendor.png")


# ---------------------------------------------------------------- Fig 4
def fig4_detection():
    """Failure detection: ROC, per-vendor external ROC, calibration."""
    s = pd.read_csv(os.path.join(RES, "qc_scores.csv"))
    perf = pd.read_csv(os.path.join(RES, "qc_performance.csv"))
    ood = pd.read_csv(os.path.join(RES, "ood_control.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.55})

    # (a) external ROC, per vendor (model never saw GE / Philips / Canon)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--", zorder=1)
    for v in VENDORS:
        g = s[s.vendor == v]
        if g["fail"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(g["fail"], g["qc_LogReg"])
        a = perf[(perf.model == "LogReg") & (perf.subgroup == v)].auroc.iloc[0]
        ax.plot(fpr, tpr, color=COL[v], lw=1.6, marker=MRK[v], markevery=0.22,
                markersize=3.5, markeredgecolor="white", markeredgewidth=0.4,
                label=f"{v}  {a:.3f}", zorder=2)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("(a) Failure detection, M&Ms\n(AUROC per vendor)", fontsize=8, pad=5)
    ax.legend(fontsize=6, loc="lower right", handlelength=1.4, labelspacing=0.25)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    # (b) the control: is it finding the vendor, or the error?
    ax = axes[1]
    dets = ["Image-OOD (Mahalanobis)", "CycleQA (disagreement, no training)", "CycleQA (LogReg)"]
    short = ["Image-OOD", "CycleQA\n(rule)", "CycleQA\n(model)"]
    tg = ["is this Canon?", "is this segmentation wrong?"]
    x = np.arange(3); w = 0.34
    for k, (t, fc, hh) in enumerate(zip(tg, ["white", "#0173B2"], ["///", ""])):
        vals = [ood[(ood.detector == d) & (ood.target == t)].auroc.iloc[0] for d in dets]
        b = ax.bar(x + (k - 0.5) * w, vals, w * 0.9, facecolor=fc, hatch=hh,
                   edgecolor=INK if fc == "white" else "white", linewidth=0.7,
                   label="vendor" if k == 0 else "error", zorder=2)
        for xi, vi in zip(x + (k - 0.5) * w, vals):
            ax.text(xi, vi + 0.02, f"{vi:.2f}", ha="center", fontsize=6, color=INK)
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=":", zorder=1)
    ax.text(2.62, 0.5, "chance", fontsize=6, color=MUTED, ha="left", va="center")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=6.6)
    ax.set_xlim(-0.62, 3.15)
    ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.12)
    ax.set_title("(b) Control: what is detected?", fontsize=8, pad=5)
    ax.legend(fontsize=6, loc="upper center", ncol=2, handlelength=1.2,
              labelspacing=0.2, columnspacing=1.0)

    # (c) calibration on the external set
    ax = axes[2]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--", zorder=1)
    pt, pp = calibration_curve(s["fail"], s["qc_LogReg"], n_bins=8, strategy="quantile")
    ax.plot(pp, pt, color="#0173B2", lw=1.6, marker="o", markersize=4.5,
            markeredgecolor="white", markeredgewidth=0.5, zorder=2)
    br = perf[(perf.model == "LogReg") & (perf.scope == "MnM (external)")
              & (perf.subgroup == "ALL")].brier.iloc[0]
    ax.text(0.96, 0.06, f"Brier = {br:.3f}", transform=ax.transAxes, ha="right",
            fontsize=7, color=INK)
    ax.set_xlabel("Predicted failure risk"); ax.set_ylabel("Observed failure rate")
    ax.set_title("(c) Calibration, M&Ms", fontsize=8, pad=5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle("The detector transfers to vendors absent from development",
                 fontsize=9, y=1.02)
    save(fig, "fig4_detection.png")


# ---------------------------------------------------------------- Fig 5
def fig5_action():
    """Acting on the risk score: selective referral and QC-guided reselection."""
    rc = pd.read_csv(os.path.join(RES, "risk_coverage.csv"))
    rs = pd.read_csv(os.path.join(RES, "reselection.csv"))
    rc = rc[rc.model == "LogReg"].sort_values("coverage")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.48})

    # (a) risk-coverage on EF error
    ax = axes[0]
    ax.plot(100 * rc.coverage, rc.residual_ef_mae, color="#0173B2", lw=1.8,
            marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=0.5, zorder=2)
    ax.set_xlabel("Coverage: studies auto-reported (%)")
    ax.set_ylabel("Residual EF error (pp)")
    ax.invert_xaxis()
    ax.set_ylim(15, 24.6)
    ax.set_title("(a) Selective referral", fontsize=8, pad=5)
    y80 = rc[rc.coverage == 0.8].residual_ef_mae.iloc[0]
    ax.annotate(f"review top 20%\n→ {y80:.1f} pp", xy=(80, y80), xytext=(72, 22.4),
                fontsize=6.2, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=MUTED))

    # (b) reselection: EF error
    ax = axes[1]
    order = ["always-bidir (current system)", "always-ES (best fixed, hindsight)",
             "CycleQA-guided reselection", "oracle: min EF error"]
    lbl = ["Current system", "Best fixed (hindsight)", "CycleQA reselection", "Oracle"]
    g = rs[rs.scope == "ALL"].set_index("strategy")
    vals = [g.loc[o, "ef_mae"] for o in order]
    fcs = ["white", "#BFBFBF", "#0173B2", "white"]
    hts = ["", "", "", "///"]
    bars = ax.bar(range(4), vals, 0.66, color=fcs, hatch=hts, edgecolor=INK,
                  linewidth=0.7, zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(lbl, fontsize=6.4, rotation=32, ha="right")
    ax.set_ylabel("EF error (pp)"); ax.set_ylim(0, 27)
    ax.set_title("(b) QC-guided reselection", fontsize=8, pad=5)

    # (c) per-vendor reselection gain in EF error
    ax = axes[2]
    x = np.arange(len(VENDORS)); w = 0.36
    cur = [rs[(rs.scope == v) & (rs.strategy == order[0])].ef_mae.iloc[0] for v in VENDORS]
    new = [rs[(rs.scope == v) & (rs.strategy == order[2])].ef_mae.iloc[0] for v in VENDORS]
    ax.bar(x - w / 2, cur, w * 0.92, facecolor="white", edgecolor=INK, linewidth=0.7,
           hatch="///", label="current system", zorder=2)
    b2 = ax.bar(x + w / 2, new, w * 0.92, facecolor="#0173B2", edgecolor="white",
                linewidth=0.5, label="+ CycleQA reselection", zorder=2)
    for xi, vi in zip(x + w / 2, new):
        ax.text(xi, vi + 0.8, f"{vi:.1f}", ha="center", fontsize=6, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(VENDORS, fontsize=6.5, rotation=20, ha="right")
    ax.set_ylabel("EF error (pp)"); ax.set_ylim(0, 40)
    ax.set_title("(c) Gain by vendor", fontsize=8, pad=5)
    ax.legend(fontsize=6, loc="upper right", handlelength=1.2, labelspacing=0.25)

    fig.suptitle("Acting on the risk score: no labels, no retraining, no extra inference",
                 fontsize=8.5, y=1.04)
    save(fig, "fig5_action.png")


if __name__ == "__main__":
    fig2_dice_estimation()
    fig3_cross_vendor()
    fig4_detection()
    fig5_action()
    print("\n[make_figures] done")
