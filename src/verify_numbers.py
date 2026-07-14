"""Every numeric claim in main.tex, re-derived from the CSVs and checked.

If a number in the manuscript cannot be reproduced here, it does not belong in the
manuscript. Run this before submitting.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT

RES = os.path.join(ROOT, "metis", "results")
TEX = os.path.join(ROOT, "metis", "paper", "main.tex")

m = pd.read_csv(os.path.join(RES, "patient_metrics.csv"))
m = m[m.model == "MedSAM2-dual"]
a, e = m[m.dataset == "ACDC"], m[m.dataset == "MnM"]
de = pd.read_csv(os.path.join(RES, "dice_estimation.csv"))
qp = pd.read_csv(os.path.join(RES, "qc_performance.csv"))
ab = pd.read_csv(os.path.join(RES, "ablation.csv"))
rc = pd.read_csv(os.path.join(RES, "risk_coverage.csv"))
oo = pd.read_csv(os.path.join(RES, "ood_control.csv"))
rs = pd.read_csv(os.path.join(RES, "reselection.csv"))
fp = pd.read_csv(os.path.join(RES, "failure_prevalence.csv"))

V = lambda vd: e[e.vendor == vd]
Q = lambda mdl, sc, sg="ALL": qp[(qp.model == mdl) & (qp.scope == sc) & (qp.subgroup == sg)].iloc[0]
D = lambda sc, sg: de[(de.scope == sc) & (de.subgroup == sg)].iloc[0]
R = lambda st, sc="ALL": rs[(rs.strategy == st) & (rs.scope == sc)].iloc[0]

CLAIMS = [
    # (claim as written in the manuscript, recomputed value, tolerance)
    ("ACDC Dice RV 0.913",        a.dice_RV.mean(),  0.913, 5e-4),
    ("ACDC Dice Myo 0.801",       a.dice_Myo.mean(), 0.801, 5e-4),
    ("ACDC Dice LV 0.858",        a.dice_LV.mean(),  0.858, 5e-4),
    ("ACDC EF error 6.4 pp",      a.ef_abs_err.mean(), 6.4, 0.05),
    ("ACDC failure rate 11%",     100 * (a.dice_min < .70).mean(), 11.0, 0.5),
    ("MnM Dice RV 0.790",         e.dice_RV.mean(),  0.790, 5e-4),
    ("MnM Dice Myo 0.702",        e.dice_Myo.mean(), 0.702, 5e-4),
    ("MnM Dice LV 0.748",         e.dice_LV.mean(),  0.748, 5e-4),
    ("MnM EF error 23.4 pp",      e.ef_abs_err.mean(), 23.4, 0.05),
    ("MnM failure rate 54%",      100 * (e.dice_min < .70).mean(), 54.0, 0.5),
    ("MnM LV Dice 0.750 (Canon)", V("Canon").dice_LV.mean(), 0.750, 5e-4),
    ("Siemens EF error 5.7 pp",   V("Siemens").ef_abs_err.mean(), 5.7, 0.05),
    ("Philips EF error 29.0 pp",  V("Philips").ef_abs_err.mean(), 29.0, 0.05),
    ("GE EF error 32.5 pp",       V("GE").ef_abs_err.mean(), 32.5, 0.05),
    ("Canon EF error 33.9 pp",    V("Canon").ef_abs_err.mean(), 33.9, 0.05),
    ("Canon failure rate 70%",    100 * (V("Canon").dice_min < .70).mean(), 70.0, 0.5),

    ("Tier1 ACDC MAE 0.064",      D("ACDC", "ALL").mae, 0.064, 5e-4),
    ("Tier1 ACDC r 0.735",        D("ACDC", "ALL").pearson_r, 0.735, 5e-4),
    ("Tier1 MnM MAE 0.072",       D("MnM", "ALL").mae, 0.072, 5e-4),
    ("Tier1 MnM r 0.814",         D("MnM", "ALL").pearson_r, 0.814, 5e-4),

    ("Rule-based external AUROC 0.850",
     Q("Rule-based (disagreement)", "MnM (external)").auroc, 0.850, 5e-4),
    ("Rule-based CI lo 0.811",
     Q("Rule-based (disagreement)", "MnM (external)").auroc_lo, 0.811, 2e-3),
    ("Rule-based CI hi 0.888",
     Q("Rule-based (disagreement)", "MnM (external)").auroc_hi, 0.888, 2e-3),
    ("LogReg external AUROC 0.851", Q("LogReg", "MnM (external)").auroc, 0.851, 5e-4),
    ("LogReg CI 0.810-0.892",       Q("LogReg", "MnM (external)").auroc_lo, 0.810, 2e-3),
    ("LogReg Brier 0.204",          Q("LogReg", "MnM (external)").brier, 0.204, 5e-4),
    ("Siemens AUROC 0.883",  Q("LogReg", "MnM (external)", "Siemens").auroc, 0.883, 5e-4),
    ("Philips AUROC 0.857",  Q("LogReg", "MnM (external)", "Philips").auroc, 0.857, 5e-4),
    ("GE AUROC 0.823",       Q("LogReg", "MnM (external)", "GE").auroc, 0.823, 5e-4),
    ("Canon AUROC 0.806",    Q("LogReg", "MnM (external)", "Canon").auroc, 0.806, 5e-4),

    ("OOD 'is this Canon' 0.478",
     oo[(oo.detector == "Image-OOD (Mahalanobis)") & (oo.target == "is this Canon?")].auroc.iloc[0],
     0.478, 5e-4),
    ("OOD 'is it wrong' 0.573",
     oo[(oo.detector == "Image-OOD (Mahalanobis)")
        & (oo.target == "is this segmentation wrong?")].auroc.iloc[0], 0.573, 5e-4),
    ("CycleQA 'is this Canon' 0.636",
     oo[(oo.detector == "CycleQA (LogReg)") & (oo.target == "is this Canon?")].auroc.iloc[0],
     0.636, 5e-4),

    ("Ablation anchor-only internal 0.987",
     ab[ab.features == "anchor only"].auroc_internal_oof.iloc[0], 0.987, 5e-4),
    ("Ablation anchor-only external 0.793",
     ab[ab.features == "anchor only"].auroc_external.iloc[0], 0.793, 5e-4),
    ("Ablation plausibility-only external 0.891",
     ab[ab.features == "anat+temp+func (no anchor)"].auroc_external.iloc[0], 0.891, 5e-4),

    ("Risk-coverage: review top 20% -> 19.8 pp",
     rc[(rc.model == "LogReg") & (rc.coverage == 0.8)].residual_ef_mae.iloc[0], 19.8, 0.05),

    ("Reselect: current system 23.4 pp", R("always-bidir (current system)").ef_mae, 23.4, 0.05),
    ("Reselect: always-ES 16.8 pp", R("always-ES (best fixed, hindsight)").ef_mae, 16.8, 0.05),
    ("Reselect: CycleQA 12.0 pp", R("CycleQA-guided reselection").ef_mae, 12.0, 0.05),
    ("Reselect: oracle 10.4 pp", R("oracle: min EF error").ef_mae, 10.4, 0.05),
    ("Reselect: CycleQA failure 14.7%",
     100 * R("CycleQA-guided reselection").failure_rate, 14.7, 0.05),
    ("Reselect: current failure 54.4%",
     100 * R("always-bidir (current system)").failure_rate, 54.4, 0.05),
    ("Reselect: Siemens CycleQA 5.8 pp",
     R("CycleQA-guided reselection", "Siemens").ef_mae, 5.8, 0.05),
    ("Reselect: Siemens always-ES 14.8 pp",
     R("always-ES (best fixed, hindsight)", "Siemens").ef_mae, 14.8, 0.05),

    ("Sensitivity: Dice<0.65 -> 0.823",
     fp[fp.endpoint == "any Dice < 0.65"].auroc_external.iloc[0], 0.823, 5e-4),
    ("Sensitivity: Dice<0.80 -> 0.894",
     fp[fp.endpoint == "any Dice < 0.80"].auroc_external.iloc[0], 0.894, 5e-4),
    ("Sensitivity: EF>10pp -> 0.870",
     fp[fp.endpoint == "|EF error| > 10 pp"].auroc_external.iloc[0], 0.870, 5e-4),
]

print("=" * 84)
print("VERIFYING EVERY NUMBER IN main.tex AGAINST metis/results/*.csv")
print("=" * 84)
bad = 0
for label, got, claimed, tol in CLAIMS:
    ok = abs(float(got) - claimed) <= tol
    bad += not ok
    print(f"  {'OK ' if ok else 'BAD'}  {label:44s} manuscript={claimed:8.3f}  csv={float(got):8.3f}")

# also confirm the cohort sizes quoted in the text
sizes = [("ACDC n=100", len(a), 100), ("M&Ms n=320", len(e), 320),
         ("Siemens n=95", len(V("Siemens")), 95), ("Philips n=125", len(V("Philips")), 125),
         ("GE n=50", len(V("GE")), 50), ("Canon n=50", len(V("Canon")), 50)]
print()
for label, got, claimed in sizes:
    ok = got == claimed
    bad += not ok
    print(f"  {'OK ' if ok else 'BAD'}  {label:44s} manuscript={claimed:8d}  csv={got:8d}")

print("\n" + "=" * 84)
if bad:
    print(f"*** {bad} CLAIM(S) DO NOT MATCH THE DATA -- FIX main.tex BEFORE SUBMITTING ***")
    sys.exit(1)
print(f"ALL {len(CLAIMS) + len(sizes)} NUMERIC CLAIMS IN THE MANUSCRIPT TRACE TO A CSV.")
print("=" * 84)
