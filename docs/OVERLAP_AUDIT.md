# Overlap audit vs the companion main-track paper (MIUA Paper 133)

Both papers appear in the MIUA 2026 proceedings. This document states exactly what is shared
and what is not. **Disclose this relationship at submission.**

## Companion paper

*Zero-Shot Full-Cycle 4D Cardiac Cine MRI Segmentation via Dual-Anchored Medical Video
Foundation Model* — introduces dual-anchored propagation and evaluates segmentation accuracy on
ACDC.

## Line-by-line

| | Paper 133 (main track) | This paper (MÉTIS) |
|---|---|---|
| **Question** | Does dual anchoring reduce propagation drift? | Does benchmark accuracy survive a change of scanner, and can the failure be seen with no ground truth? |
| **Contribution** | Dual-anchored propagation | CycleQA: no-reference failure detection + label-free reselection |
| **Role of the model** | Object of study | **Fixed upstream system**, cited, not modified |
| **Data** | ACDC (100), Siemens only | **M&Ms (320): 4 vendors, 5 centres, 9 pathologies** — never reported before. ACDC is used only as the development set. |
| **Role of ground truth** | Measures segmentation accuracy | **Only defines and evaluates the endpoint.** Never an input. |
| **Metrics** | Dice, HD95, ASSD | AUROC, AUPRC, Brier, calibration, risk–coverage, Dice-estimation MAE |
| **Figures** | methods, qualitative, time-volume, Dice boxplot, pathology heatmap | **4 entirely new figures** |
| **Tables** | segmentation, biomarkers, bbox noise, clinical | **3 entirely new tables** |
| **Conclusion** | Dual anchoring propagates more stably | Zero-shot FMs are vendor-overfitted; the failure is detectable without labels, and actionable |

## What is shared

Only the **description of the upstream system** (Methods §3.1, ~1 paragraph), which is
necessary to state what is being audited, and which cites `li2026dualanchor` explicitly.

## What is NOT reproduced

No result, figure, table, or headline number from Paper 133 appears here. The ACDC Dice figures
quoted in our abstract (0.913/0.801/0.858) are **recomputed over all 100 patients**; Paper 133
reports a 20-patient validation subset (0.850/0.809/0.843). Different cohort, different numbers.

## The intellectual link (and why this is an extension, not a repackaging)

Paper 133 introduced dual anchoring **to make the model accurate**. This paper shows the same
redundancy **also makes it self-auditing** — the two propagation directions it created are the
free ensemble that CycleQA runs on. Before that paper, this signal did not exist.
