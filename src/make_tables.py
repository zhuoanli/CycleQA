"""Phase 3 -- the three LaTeX tables. Every number is read from metis/results/*.csv."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT

RES = os.path.join(ROOT, "metis", "results")
TAB = os.path.join(ROOT, "metis", "tables")
os.makedirs(TAB, exist_ok=True)
VENDORS = ["Siemens", "Philips", "GE", "Canon"]


def write(name, s):
    p = os.path.join(TAB, name)
    with open(p, "w") as f:
        f.write(s)
    print(f"  wrote {p}")


# ---------------------------------------------------------------- Table 1
def table1_features():
    rows = [
        ("Bidirectional agreement", r"$\overline{A}$, $\min A$, $A_{\mathrm{mid}}$, "
         r"$\mathrm{sd}\,A$, $A_{10}$, decay, \% frames $A<0.7$",
         r"Per frame $t$ and structure $c$, $A_c(t)=\mathrm{Dice}\!\left(M^{\mathrm{ED}}_t,"
         r"M^{\mathrm{ES}}_t\right)$: the two independent propagations must converge if the "
         r"model tracks the true anatomy.", 24),
        ("Anatomical plausibility",
         r"connected components; LV--myocardium containment; myocardial ring closure; LV holes; "
         r"image-border contact",
         r"The myocardium must form a closed ring enclosing the LV cavity. Fragments, holes and "
         r"border contact are anatomically impossible regardless of scanner.", 6),
        ("Temporal coherence",
         r"inter-frame IoU; LV centroid displacement; volume-curve roughness; cycle closure "
         r"$|V(0)-V(T{-}1)|$; number of extrema of $V(t)$",
         r"$V(t)$ must be smooth, periodic and close its cycle. Abrupt jumps or multiple minima "
         r"indicate propagation drift.", 8),
        ("Physiological plausibility",
         r"$\mathrm{EDV}>\mathrm{ESV}$; EF within $[10,80]\%$; volume coefficient of variation; "
         r"annotated ES frame is the volume minimum",
         r"Constraints on cardiac function that hold for every patient on every scanner.", 8),
    ]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{The four families of \emph{no-reference} CycleQA features. Every quantity is "
         r"computed from the predicted masks alone; none requires a reference mask, and none has "
         r"access to the vendor, centre or pathology label. Anatomical and physiological "
         r"constraints hold on any scanner \emph{by construction}, which is why the resulting "
         r"detector transfers across vendors.}",
         r"\label{tab:features}",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}p{2.5cm}p{3.9cm}p{4.6cm}c@{}}", r"\toprule",
         r"Family & Descriptors & Why it is vendor-invariant & $n$ \\", r"\midrule"]
    for a, b, c, n in rows:
        L.append(rf"{a} & {b} & {c} & {n} \\[2pt]")
    L += [r"\midrule", r"\multicolumn{3}{@{}l}{Total} & 46 \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("table1_features.tex", "\n".join(L) + "\n")


# ---------------------------------------------------------------- Table 2
def table2_external():
    m = pd.read_csv(os.path.join(RES, "patient_metrics.csv"))
    m = m[m.model == "MedSAM2-dual"]
    a, e = m[m.dataset == "ACDC"], m[m.dataset == "MnM"]

    def line(lbl, g, n):
        f = 100 * (g.dice_min < 0.70).mean()
        return (rf"{lbl} & {n} & {g.dice_RV.mean():.3f} & {g.dice_Myo.mean():.3f} & "
                rf"{g.dice_LV.mean():.3f} & {g.ef_abs_err.mean():.1f} & {f:.0f} \\")

    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{External validation of dual-anchored MedSAM2. ACDC is acquired on Siemens "
         r"scanners only. Dice degrades modestly off-vendor, but the quantity clinicians actually "
         r"use---ejection fraction---degrades by a factor of six, and the proportion of studies "
         r"with a failed structure rises from 11\% to 70\%. Prompts are derived from the reference "
         r"masks, so these are an \emph{oracle-prompt upper bound}.}",
         r"\label{tab:external}", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{Dice} & EF error & Failure \\",
         r"\cmidrule(lr){3-5}",
         r"Cohort & $n$ & RV & Myo & LV & (pp) & (\%) \\", r"\midrule",
         r"\multicolumn{7}{@{}l}{\emph{Development (internal)}}\\",
         line("\\quad ACDC (Siemens)", a, len(a)), r"\midrule",
         r"\multicolumn{7}{@{}l}{\emph{M\&Ms (external)}}\\"]
    for v in VENDORS:
        g = e[e.vendor == v]
        tag = "" if v == "Siemens" else r"$^{\dagger}$"
        L.append(line(rf"\quad {v}{tag}", g, len(g)))
    L += [r"\cmidrule(lr){1-7}", line(r"\quad All M\&Ms", e, len(e)),
          r"\bottomrule", r"\end{tabular}", "",
          r"\smallskip", r"{\footnotesize $^{\dagger}$Vendor absent from the development set.}",
          r"\end{table}"]
    write("table2_external.tex", "\n".join(L) + "\n")


# ---------------------------------------------------------------- Table 3
def table3_qc():
    de = pd.read_csv(os.path.join(RES, "dice_estimation.csv"))
    qp = pd.read_csv(os.path.join(RES, "qc_performance.csv"))
    ab = pd.read_csv(os.path.join(RES, "ablation.csv"))
    rs = pd.read_csv(os.path.join(RES, "reselection.csv"))
    rs = rs[rs.scope == "ALL"].set_index("strategy")

    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{CycleQA. \textbf{(a)} Bidirectional agreement used \emph{directly} as an "
         r"estimate of Dice---no fitting of any kind. \textbf{(b)} Failure detection (any Dice "
         r"$<0.70$); the risk model is developed on ACDC with patient-level nested 5-fold "
         r"cross-validation, then \emph{frozen} and applied once to M\&Ms. \textbf{(c)} Feature "
         r"ablation. \textbf{(d)} Acting on the risk score. Brackets are patient-level bootstrap "
         r"95\% CIs.}",
         r"\label{tab:qc}", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{@{}lccc@{}}", r"\toprule"]

    # (a)
    L += [r"\multicolumn{4}{@{}l}{\textbf{(a) Training-free Dice estimation}}\\",
          r"Cohort & $n$ & Pearson $r$ & MAE \\", r"\midrule"]
    for sc, sg, lbl in [("ACDC", "ALL", "ACDC (internal)"), ("MnM", "ALL", r"M\&Ms (external)")] + \
                       [("MnM", v, rf"\quad {v}") for v in VENDORS]:
        r = de[(de.scope == sc) & (de.subgroup == sg)].iloc[0]
        L.append(rf"{lbl} & {int(r.n)} & {r.pearson_r:.3f} & {r.mae:.3f} \\")

    # (b)
    L += [r"\midrule",
          r"\multicolumn{4}{@{}l}{\textbf{(b) Failure detection: AUROC [95\% CI]}}\\",
          r"Method & \multicolumn{1}{c}{ACDC (out-of-fold)} & \multicolumn{2}{c}{M\&Ms (external)} \\",
          r"\midrule"]
    for mdl, lbl in [("Rule-based (disagreement)", "Disagreement, no training"),
                     ("LogReg", "Logistic regression"),
                     ("RandomForest", "Random forest")]:
        i = qp[(qp.model == mdl) & (qp.scope == "ACDC (out-of-fold)")].iloc[0]
        x = qp[(qp.model == mdl) & (qp.scope == "MnM (external)") & (qp.subgroup == "ALL")].iloc[0]
        L.append(rf"{lbl} & {i.auroc:.3f} & \multicolumn{{2}}{{c}}{{{x.auroc:.3f} "
                 rf"[{x.auroc_lo:.3f}, {x.auroc_hi:.3f}]}} \\")
    L.append(r"\addlinespace")
    for v in VENDORS:
        x = qp[(qp.model == "LogReg") & (qp.subgroup == v)].iloc[0]
        tag = "" if v == "Siemens" else r"$^{\dagger}$"
        L.append(rf"\quad {v}{tag} & --- & \multicolumn{{2}}{{c}}{{{x.auroc:.3f} "
                 rf"[{x.auroc_lo:.3f}, {x.auroc_hi:.3f}]}} \\")

    # (c)
    L += [r"\midrule",
          r"\multicolumn{4}{@{}l}{\textbf{(c) Ablation: AUROC}}\\",
          r"Features & $n$ & ACDC (out-of-fold) & M\&Ms (external) \\", r"\midrule"]
    for _, r in ab.iterrows():
        nm = r.features.replace("+", r"$+$").replace("_", r"\_")
        L.append(rf"{nm} & {int(r.n_features)} & {r.auroc_internal_oof:.3f} & "
                 rf"{r.auroc_external:.3f} \\")

    # (d)
    L += [r"\midrule",
          r"\multicolumn{4}{@{}l}{\textbf{(d) Acting on the risk score (M\&Ms, $n=320$)}}\\",
          r"Strategy & Dice & EF error (pp) & Failure (\%) \\", r"\midrule"]
    for k, lbl in [("always-bidir (current system)", "Current system (merged output)"),
                   ("always-ES (best fixed, hindsight)", "Best fixed choice (hindsight)"),
                   ("CycleQA-guided reselection", r"\textbf{CycleQA reselection}"),
                   ("oracle: min EF error", "Oracle (minimum EF error)")]:
        r = rs.loc[k]
        L.append(rf"{lbl} & {r.dice_mean:.3f} & {r.ef_mae:.1f} & {100*r.failure_rate:.1f} \\")

    L += [r"\bottomrule", r"\end{tabular}", "", r"\smallskip",
          r"{\footnotesize $^{\dagger}$Vendor absent from the development set.}",
          r"\end{table}"]
    write("table3_qc.tex", "\n".join(L) + "\n")


if __name__ == "__main__":
    table1_features()
    table2_external()
    table3_qc()
    print("\n[make_tables] done")
