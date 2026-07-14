# CycleQA

**Cross-Vendor Quality Assurance of Zero-Shot Full-Cycle 4D Cardiac Cine MRI Segmentation
via Dual-Anchored Medical Video Foundation Model**

Zhuoan Li — MIUA 2026 (MÉTIS Special Session)

Everything here is self-contained: preprocessing → segmentation inference → quality assurance →
figures and tables. Nothing was typed by hand; every number in the paper is re-derived from a CSV
by `src/verify_numbers.py`.

---

## The problem

A medical video foundation model needs no labels to **run**. So it has no labels with which to be
**checked**.

Dual-anchored MedSAM2 segments the full cardiac cycle zero-shot — no training, two box prompts —
and scores **0.913 / 0.801 / 0.858** Dice (RV/Myo/LV) on ACDC. But ACDC is one hospital on
**Siemens** scanners. We validated the same system on **320 M&Ms patients** across four
manufacturers.

The naive split (*development vs external*) confounds three things at once — new dataset, new
hospital, new manufacturer. Separating them is the whole point:

| | What changed | Mean Dice | **EF error** | **Failed studies** |
|---|---|---|---|---|
| **ACDC** (Siemens) | — (development) | 0.857 | **6.4 pp** | **11%** |
| **M&Ms Siemens** | new dataset, new hospital, **same scanner** | 0.817 | **5.7 pp** | 28% |
| **M&Ms Philips** | **new manufacturer** | 0.705 | **29.0 pp** | 66% |
| **M&Ms GE** | **new manufacturer** | 0.746 | **32.5 pp** | 58% |
| **M&Ms Canon** | **new manufacturer** | 0.718 | **33.9 pp** | 70% |

> ### A new hospital on the same scanner costs nothing. A new scanner costs everything.

And notice what Dice does versus what EF does. **Dice degrades modestly — from 0.86 to 0.70–0.75.
A reader looking only at Dice would call the system "degraded but usable." The ejection fraction
computed from those same masks is off by 34 percentage points, which is clinically worthless.**
An EF error of 34 pp turns a healthy heart (60%) into reported severe heart failure (26%).

**Benchmark Dice conceals the clinical damage. And at that hospital there is no ground truth with
which to notice.** The model fails silently.

## The idea

Dual anchoring propagates the same video **twice**: forward from the end-diastolic anchor
(`ed_pred`), backward from the end-systolic anchor (`es_pred`). Both span every frame. Both are
already written to disk. Nobody had looked at the relationship *between* them.

> **If the model tracks the true anatomy, the two independent trajectories converge.
> If it drifts, they separate. The magnitude of the disagreement is the magnitude of the error.**

That is CycleQA — after the cardiac cycle that must close. It needs **no reference mask, no second
network, no ensemble, no registration, no retraining, and no extra inference.**

A frame-wise CNN **cannot produce this signal** — with no propagation there is no second direction
to disagree. That is why prior reference-free CMR quality-control work, all of which targets
frame-wise supervised models, could not have found it.

## The results

**Estimate segmentation quality with no ground truth and no training at all:**

| | Pearson *r* | MAE |
|---|---|---|
| ACDC (development) | 0.735 | 0.064 |
| **M&Ms (external)** | **0.814** | **0.072** |

The signal is *stronger* out of domain than in it — it is most informative exactly where it is
needed.

**Detect failed studies.** The risk model is fitted on ACDC, **frozen**, and applied once to M&Ms.
It never sees a Philips, GE or Canon study during development:

AUROC **0.850** [0.811 – 0.888] — Siemens 0.883 · Philips 0.857 · GE 0.823 · **Canon 0.806**.

**It detects the error, not the scanner.** A detector that merely recognises unusual-looking images
would be useless: it cannot separate *"unusual-looking but correct"* from *"broken"*. So every
detector is scored against **two different questions**:

| | *Is this an unusual scanner?* | *Is the segmentation wrong?* |
|---|---|---|
| Looks at the **image** (appearance OOD) | 0.48 | 0.57 |
| Looks at the **mask** (CycleQA) | 0.64 | **0.85** |

**And it fixes things.** Three predictions per patient are already on disk. Which is best is
domain-dependent — and you have no labels to choose. CycleQA scores each and keeps the most
plausible:

| Strategy | Mean Dice | **EF error** | **Failure rate** |
|---|---|---|---|
| **Today** (report the merged output) | 0.747 | **23.4 pp** | **54.4%** |
| Best fixed choice (*needs labels to know*) | 0.799 | 16.8 pp | 21.9% |
| **CycleQA reselection** | **0.815** | **12.0 pp** | **14.7%** |
| Oracle (upper bound) | 0.812 | 10.4 pp | 17.8% |

**Zero labels. Zero retraining. Zero extra inference.**

On Siemens it keeps the merged / ED-anchored output (EF error 5.8 pp) — whereas *always* choosing
ES, the best global strategy, would have wrecked it (14.8 pp). On Philips, GE and Canon it switches
to ES. **No fixed strategy works everywhere. CycleQA adapts per patient and per manufacturer,
without ever being told which manufacturer it is looking at.**

## Why quality assurance transfers when segmentation does not

The segmenter must answer a **hard** question — *which pixels are myocardium?* — and to do that it
must read **appearance**. Appearance is exactly what changed. So it broke.

CycleQA asks two **easy** questions — *are you consistent with yourself?* and *is your answer
anatomically possible?* Neither requires the image. **A heart is a heart on any scanner:** the
myocardium encloses the LV cavity, the volume curve closes, EDV exceeds ESV. These constraints are
vendor-invariant **by construction**.

> **Verifying an answer is easier than producing it. That is why the watchdog is more robust than
> the model it watches.**

---

## Reproduce

The quality-assurance stage needs **no GPU, no training and no inference** — it reads the
predictions produced by the pipeline. ~20 min on 48 CPU cores.

```bash
conda activate cinema_ft            # numpy, scipy, pandas, scikit-learn, matplotlib
bash jobs/run_all.sh                # verify → metrics → features → QC → figures → tables
python src/verify_numbers.py        # all 54 manuscript numbers re-derived from the CSVs
bash jobs/watch.sh                  # live progress
```

To regenerate the predictions from scratch, see **[`docs/PIPELINE.md`](docs/PIPELINE.md)** —
preprocessing and inference code is in `pipeline/`.

## Layout

```
pipeline/   prep_acdc_4d.py       raw 4D cine + ED/ES masks -> 512x512 NPZ, label convention harmonised
            prep_mnm.py           same, for the multi-vendor cohort
            infer_medsam2.py      dual-anchored propagation -> ed_pred / es_pred / bidir
            infer_sam2.py         the non-medical SAM2 baseline

src/        verify_data.py        BLOCKING gate. Rules out an ED/ES indexing artefact; re-derives
                                  the label convention geometrically. Exits non-zero on failure.
            compute_metrics.py    Per-patient Dice and EF. THE ONLY PLACE GROUND TRUTH DEFINES
                                  THE ENDPOINT.
            extract_features.py   46 no-reference features. ASSERTS no ground truth, manufacturer,
                                  centre or pathology label can enter the matrix.
            variant_features.py   The same features per prediction variant — needed to choose between them.
            train_qc.py           Nested CV -> frozen -> external. Plus the appearance-OOD control,
                                  risk–coverage, ablation, threshold sensitivity.
            reselect.py           Score all three predictions, keep the most plausible.
            make_figures.py       4 figures.      make_tables.py   3 LaTeX tables.
            verify_numbers.py     Every number in the paper, re-derived. Non-zero exit on mismatch.

docs/       PIPELINE   end-to-end, including how EF is computed and why it is trustworthy
            DATASETS   cohorts, label convention, and the verification gate
            MODELS     what each prediction directory is, and what is excluded and why
            SKILLS     leakage control, nested CV, calibration, selective prediction — and the why
            ENVIRONMENT, RESULTS_SUMMARY

paper/      main.tex (LNCS) + references.bib   metispaper.md explains every section, figure and table
```

## Honesty notes

These are in the paper. They belong here too.

- **Prompts are derived from the reference masks.** Every number is an **oracle-prompt upper
  bound**. A deployed system additionally needs automatic prompt generation. This makes the finding
  *stronger*: even with perfect prompts, the model breaks on an unseen manufacturer.
- **Reference masks exist only at ED and ES.** We validate annotated-phase error and claim nothing
  about intermediate phases. Full-cycle features are *inputs*, never validated targets.
- **Learning is not what makes this work.** Anchor features fitted with logistic regression score
  0.987 internally but **0.793** externally — textbook overfitting on 100 patients containing 11
  failures. The **untrained** rule transfers *better* (**0.850**). This is reported rather than
  hidden: the disagreement signal is strong enough that no training is required, and on a
  development set this small, training on it is actively harmful.
- **Retrospective, public data. No prospective clinical validation.**
- Supervised U-Net and DINOv2 predictions exist for the external cohort, but their checkpoint
  provenance could not be established, so they are excluded rather than defended. See
  [`docs/MODELS.md`](docs/MODELS.md).

## Datasets

Both are public and are not redistributed here.

- **ACDC** — Bernard et al. 2018, `10.1109/TMI.2018.2837502`. 100 patients, 1 centre, Siemens.
- **M&Ms** — Campello et al. 2021, `10.1109/TMI.2021.3090082`. 320 patients, 4 manufacturers,
  5 centres, 9 pathologies.

## Citation

```bibtex
@inproceedings{li2026cycleqa,
  title     = {Cross-Vendor Quality Assurance of Zero-Shot Full-Cycle 4D Cardiac Cine MRI
               Segmentation via Dual-Anchored Medical Video Foundation Model},
  author    = {Li, Zhuoan},
  booktitle = {Medical Image Understanding and Analysis (MIUA)},
  year      = {2026}
}
```
