# Methods and techniques — what was used, and why

A reader (or an interviewer) should be able to challenge any choice here. Each entry says what
the technique is, why it was necessary, and where it lives.

## Leakage control — the thing that makes or breaks a QC paper

**Problem.** A quality-control model that peeks at ground truth is worthless: at deployment
there is none. It is very easy to leak by accident.

**What we do.**
- `src/extract_features.py` asserts, before writing, that no feature column matches
  `gt_`, `true_`, `dice_`, `hd95`, `assd`, `vendor`, `centre`, `pathology`, `ef_abs_err`.
  This assertion **fired in practice** on a feature named `func_edv_gt_esv` — where `gt_` meant
  "greater than," not "ground truth." It was renamed. A strict check that false-positives is
  working correctly.
- Ground truth appears in exactly two files: `compute_metrics.py` (to define the endpoint) and
  `train_qc.py` (to evaluate against it). Never as an input.
- Vendor, centre and pathology are used **only** for subgroup reporting, never as features. The
  point is that the detector transfers across vendors *without being told the vendor*.

## Nested cross-validation

**Problem.** Fitting a scaler or selecting a hyperparameter on the full development set leaks
test information into training, and inflates the reported score.

**What we do.** `train_qc.py::nested_cv_oof` — an outer 5-fold split for reporting, an inner
4-fold `GridSearchCV` for hyperparameters. Imputation, scaling and model selection all happen
*inside* the training fold. Only out-of-fold predictions are ever reported. Training-set
performance is never printed.

## Frozen external validation

The point of the paper. The risk model is fitted on ACDC, **frozen**, and applied exactly once
to M&Ms. It is never refit, never tuned on the external data, and never sees a Philips, GE or
Canon study during development. Per-vendor AUROC on M&Ms is therefore a genuine
out-of-distribution generalisation measurement, not a cross-validation score.

## Bootstrap confidence intervals

2000 patient-level resamples (`train_qc.py::boot_ci`). Resampling is at the *patient* level, not
the slice level — slices within a patient are not independent, and resampling them would give
CIs that are far too narrow.

## Calibration

AUROC is rank-based and says nothing about whether a "0.8 risk" case actually fails 80% of the
time. For a triage system that matters, so we report the Brier score and a calibration curve
(`guo2017calibration`).

## Risk–coverage / selective prediction

`geifman2017selective`. Rank studies by risk, refer the top X% to human review, and report the
residual error among the studies that were auto-reported. This converts "the model is
sometimes wrong" into "here is what a 20%-review budget buys you," which is the form a clinical
site can actually act on.

## Threshold sensitivity — pre-specified, not tuned

The failure endpoint (any Dice < 0.70) is a modelling choice. Choosing the threshold that
maximises AUROC would be a form of cherry-picking. The threshold is pre-specified and then
**swept** (Dice 0.65–0.80; EF error 5–15 pp) with all results reported; `train_qc.py` writes them
to `results/failure_prevalence.csv`.

## The control experiment

A reviewer will ask: "isn't CycleQA just a Canon detector?" Answering requires evaluating every
detector against **two different targets** — *is this Canon?* and *is this segmentation wrong?*
CycleQA scores 0.636 on the first and 0.851 on the second: it is an error detector. See
`train_qc.py::ood_control`.

## Reproducibility

`src/verify_numbers.py` re-derives all 54 numeric claims in the manuscript from the CSVs and
exits non-zero on any mismatch. No number in the paper was typed by hand.

## Visualisation

Figures are print-safe for LNCS, which is often read in greyscale: vendor identity is carried by
**marker shape and hatch as well as hue**, so colour never encodes identity alone. The
four-vendor palette was validated for lightness band, chroma floor, adjacent-pair
colour-vision-deficiency separation (≥12 ΔE across deuteran/protan/tritan) and ≥3:1 contrast on
white.
