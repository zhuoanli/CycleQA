# CycleQA

**Cross-Vendor Quality Assurance of Zero-Shot Full-Cycle 4D Cardiac Cine MRI Segmentation
via Dual-Anchored Medical Video Foundation Model**

Zhuoan Li, Behzad Sharif — MIUA 2026 (MÉTIS Special Session)

---

## The problem

A medical video foundation model needs no labels to *run*. So it has no labels with which to be
*checked*.

Dual-anchored MedSAM2 segments the full cardiac cycle zero-shot and scores **0.913/0.801/0.858**
Dice on ACDC. But ACDC is one hospital, one vendor (Siemens). We validated it on **320
previously unreported M&Ms patients** — 4 vendors, 5 centres, 9 pathologies:

| | Dice (RV/Myo/LV) | **EF error** | **Failed studies** |
|---|---|---|---|
| ACDC (Siemens) | 0.913 / 0.801 / 0.858 | **6.4 pp** | **11%** |
| M&Ms (4 vendors) | 0.790 / 0.702 / 0.748 | **23.4 pp** | **54%** |
| — Canon | 0.751 / 0.653 / 0.750 | **33.9 pp** | **70%** |

Dice degrades *modestly*. The number a cardiologist actually reads — ejection fraction — degrades
**sixfold**. A 34-point EF error is clinically unusable, and **at that hospital there is no
ground truth with which to notice.** The model fails silently.

## The idea

Dual anchoring propagates the same video **twice**: forward from the ED anchor (`ed_pred`),
backward from the ES anchor (`es_pred`). Both span every frame. Both are already on disk. Nobody
had looked at the relationship *between* them.

> **If the model tracks the true anatomy, the two independent trajectories converge.
> If it drifts, they separate. The magnitude of the disagreement is the magnitude of the error.**

That is CycleQA — after the cardiac cycle that must close. It needs **no reference mask, no
second model, no ensemble, no registration, no retraining, and no extra inference.**

## Results

**Estimate Dice with no ground truth and no training at all:**

| | Pearson r | MAE |
|---|---|---|
| ACDC | 0.735 | 0.064 |
| **M&Ms (external)** | **0.814** | **0.072** |

The signal is *stronger* out of domain — most informative exactly where it is needed.

**Detect failed studies** (model frozen on ACDC, applied once to M&Ms):

AUROC **0.850** [0.811–0.888] — Siemens 0.883 · Philips 0.857 · GE 0.823 · **Canon 0.806**
(a manufacturer the model never saw).

**It detects the error, not the vendor:**

| | *"is this Canon?"* | *"is this segmentation wrong?"* |
|---|---|---|
| Image-based OOD | 0.478 | 0.573 |
| **CycleQA** | 0.636 | **0.851** |

**And it fixes things.** The system already wrote three predictions per patient. Which is best is
domain-dependent — and you have no labels to choose. CycleQA scores each and keeps the most
plausible:

| Strategy | Dice | **EF error** | **Failure rate** |
|---|---|---|---|
| Current system | 0.747 | **23.4 pp** | **54.4%** |
| Best fixed choice (hindsight only) | 0.799 | 16.8 pp | 21.9% |
| **CycleQA reselection** | **0.815** | **12.0 pp** | **14.7%** |
| Oracle | 0.812 | 10.4 pp | 17.8% |

**Zero labels. Zero retraining. Zero extra inference.** On Siemens it keeps the merged output
(5.8 pp); on Canon/GE/Philips it switches to ES-anchored. **It adapts per patient and per domain
without ever being told the vendor.**

## Why quality assurance transfers when segmentation does not

The segmenter must answer a hard question — *which pixels are myocardium?* — and to do that it
must read **appearance**. Appearance is what changed. So it broke.

CycleQA asks two easy questions — *are you consistent with yourself?* and *is your answer
anatomically possible?* Neither needs the image. **A heart is a heart on any scanner:** the
myocardium encloses the LV cavity, the volume curve closes, EDV exceeds ESV. These constraints
are vendor-invariant *by construction*.

**Verifying an answer is easier than producing it. The watchdog is more robust than the model it
watches.**

## Reproduce

CPU only. No GPU, no training, no inference — all predictions already exist. ~20 min on 48 cores.

```bash
module load conda && conda activate cinema_ft
bash metis/jobs/run_all.sh          # verify -> metrics -> features -> QC -> figures -> tables
python metis/src/verify_numbers.py  # all 54 manuscript numbers re-derived from the CSVs
```

`bash metis/jobs/watch.sh` shows live progress.

## Layout

```
src/     verify_data.py      Gate 0a — blocking. Rules out an ED/ES indexing artefact.
         compute_metrics.py  Gate 0b — reference metrics. THE ONLY PLACE GT DEFINES THE ENDPOINT.
         extract_features.py Phase 1 — 46 no-reference features. Asserts no GT leaks in.
         variant_features.py Phase 2 — per-variant features, for reselection.
         train_qc.py         Phase 2 — nested CV, frozen external validation, OOD control,
                                       risk–coverage, ablation, threshold sensitivity.
         reselect.py         Phase 2 — QC-guided reselection.
         make_figures.py     Phase 3 — 4 figures. make_tables.py — 3 LaTeX tables.
         verify_numbers.py   Every number in the paper, re-derived. Exits non-zero on mismatch.
docs/    DATASETS · MODELS · ENVIRONMENT · SKILLS · OVERLAP_AUDIT · RESULTS_SUMMARY
paper/   main.tex (LNCS) + references.bib (53 entries; every new one DOI/arXiv-verified)
```

## Honesty notes

These are in the paper. They belong here too.

- **Prompts are derived from the reference masks.** All numbers are an **oracle-prompt upper
  bound**. A real deployment additionally needs automatic prompt generation.
- **Reference masks exist only at ED and ES.** We validate annotated-phase error and claim
  nothing about intermediate phases. Full-cycle features are *inputs*, never validated targets.
- **Learning is not what makes this work.** Anchor features fitted with logistic regression score
  0.987 internally but only 0.793 externally — textbook overfitting on 100 patients with 11
  failures. The **untrained** rule transfers *better* (0.850). We report this rather than hide
  it: the disagreement signal is strong enough that no training is required.
- **Retrospective, public data, no prospective clinical validation.**
- U-Net and DINOv2 M&Ms predictions exist but their checkpoint provenance is unresolved, so they
  are excluded rather than defended. See `docs/MODELS.md`.

## Relation to the companion paper

This extends our MIUA main-track paper (*Zero-Shot Full-Cycle 4D Cardiac Cine MRI Segmentation
via Dual-Anchored Medical Video Foundation Model*), which introduces dual-anchored propagation
and evaluates its accuracy on ACDC. Here that system is **fixed**, and the question is different:
no-reference failure detection under distribution shift, on 320 patients never reported before.
No result, figure or table of the companion paper is reproduced. See `docs/OVERLAP_AUDIT.md`.

**Paper 133 introduced dual anchoring to make the model accurate. This paper shows the same
redundancy also makes it self-auditing.**
