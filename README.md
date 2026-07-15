# CycleQA

**Cross-Vendor Quality Assurance of Zero-Shot Full-Cycle 4D Cardiac Cine MRI Segmentation
via Dual-Anchored Medical Video Foundation Model**

Zhuoan Li — MIUA 2026 (MÉTIS Special Session)

Code for every result in the paper: preprocessing → segmentation inference → quality assurance →
figures and tables. The method in one picture:

![CycleQA method overview](figures/method.png)
 **No number in the manuscript was typed by hand** — `src/verify_numbers.py`
re-derives all 54 of them from the CSVs and exits non-zero on any mismatch.

---

## The problem

A medical video foundation model needs no labels to **run**. So it has no labels with which to be
**checked**.

Dual-anchored MedSAM2 segments the full cardiac cycle zero-shot — no training, two box prompts —
and scores **0.913 / 0.801 / 0.858** Dice (RV / Myo / LV) on ACDC. But ACDC is one hospital on
**Siemens** scanners. We validated the same system on **320 M&Ms patients** across four
manufacturers.

The naive comparison (*development vs external*) confounds three changes at once — new dataset,
new hospital, new manufacturer. Separating them is the whole point:

| Cohort | What changed | Mean Dice | **EF error** | **Failed studies** |
|---|---|---|---|---|
| **ACDC** (Siemens) | — (development) | 0.857 | **6.4 pp** | **11 %** |
| **M&Ms Siemens** | new dataset, new hospital, **same scanner** | 0.817 | **5.7 pp** | 28 % |
| **M&Ms Philips** | **new manufacturer** | 0.705 | **29.0 pp** | 66 % |
| **M&Ms GE** | **new manufacturer** | 0.746 | **32.5 pp** | 58 % |
| **M&Ms Canon** | **new manufacturer** | 0.718 | **33.9 pp** | 70 % |

> ### A new hospital on the same scanner costs nothing. A new scanner costs everything.

And look at what **Dice** does versus what **EF** does. Dice degrades only modestly — 0.86 → 0.70–0.75.
A reader seeing that would call the system *"degraded but usable."* The ejection fraction computed
from **those same masks** is off by 34 percentage points. An EF error of 34 pp turns a healthy heart
(60 %) into reported severe heart failure (26 %).

**Benchmark Dice conceals the clinical damage — and at that hospital there is no ground truth with
which to notice.** The model fails silently.

*(“pp” = percentage point. EF is itself a percentage, so an error of “6.4 pp” means the estimate was
6.4 points off — writing “6.4 %” would ambiguously suggest a relative error.)*

## The idea

Dual anchoring propagates the same video **twice**: once anchored at the end-diastolic frame
(`ed_pred`), once at the end-systolic frame (`es_pred`). Each anchor propagates in *both* temporal
directions, so **both predictions span every frame of the cycle** -- including the frames before ED
and after ES. **Both are
already written to disk.** Nobody had looked at the relationship *between* them.

> **If the model tracks the true anatomy, the two independent trajectories converge.
> If it drifts, they separate. The magnitude of the disagreement is the magnitude of the error.**

Concretely, for frame *t* and structure *c*:

```
A_c(t) = Dice( ed_pred[t] , es_pred[t] )          # agreement, no ground truth anywhere
disagreement map = ( ed_pred[t] != es_pred[t] )   # the voxels the two passes label differently
```

That is CycleQA — after the cardiac cycle that must close. It needs **no reference mask, no second
network, no ensemble, no registration, no retraining, and no extra inference.**

A frame-wise CNN **cannot produce this signal**: with no propagation there is no second direction to
disagree. This is why prior reference-free CMR quality-control work — all of which targets
frame-wise supervised models — could not have found it.

## The results

**Estimate segmentation quality with no ground truth and no training at all:**

| | Pearson *r* | MAE |
|---|---|---|
| ACDC (development) | 0.735 | 0.064 |
| **M&Ms (external)** | **0.814** | **0.072** |

The signal is *stronger* out of domain than in it — it is most informative exactly where it is
needed.

**Detect failed studies.** The risk model is fitted on ACDC with patient-level nested
cross-validation, **frozen**, and applied once to M&Ms. It never sees a Philips, GE or Canon study
during development:

AUROC **0.850** [0.811 – 0.888] — Siemens 0.883 · Philips 0.857 · GE 0.823 · **Canon 0.806**.

**It detects the error, not the scanner.** A detector that merely recognises unusual-looking images
would be useless — it cannot separate *"unusual-looking but correct"* from *"broken"*. So every
detector is scored against **two different targets**:

| | *unseen manufacturer* | *failed segmentation* |
|---|---|---|
| Reads the **image** (appearance OOD) | 0.48 | 0.57 |
| Reads the **mask** (CycleQA) | 0.64 | **0.85** |

**And it fixes things.** Three predictions per patient are already on disk. Which is best is
domain-dependent — and you have no labels with which to choose. CycleQA scores each and keeps the
most plausible:

| Strategy | Mean Dice | **EF error** | **Failure rate** |
|---|---|---|---|
| **Current system** (report the merged output) | 0.747 | **23.4 pp** | **54.4 %** |
| Best fixed choice (*needs labels to identify*) | 0.799 | 16.8 pp | 21.9 % |
| **CycleQA reselection** | **0.815** | **12.0 pp** | **14.7 %** |
| Oracle (upper bound) | 0.812 | 10.4 pp | 17.8 % |

**Zero labels. Zero retraining. Zero extra inference.**

On Siemens it keeps the merged / ED-anchored output (EF error 5.8 pp) — whereas *always* choosing ES,
the best global strategy, would have wrecked it (14.8 pp). On Philips, GE and Canon it switches to
ES. **No fixed strategy works everywhere. CycleQA adapts per patient and per manufacturer, without
ever being told which manufacturer it is looking at.**

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

## Reproducing every result in the paper

### Requirements

```bash
conda env create -f environment.yml     # or: pip install -r requirements.txt
conda activate cycleqa
```
Python ≥ 3.9 · numpy · scipy · pandas · scikit-learn · matplotlib · nibabel.
PyTorch + the MedSAM2 checkpoint are needed **only** to regenerate the predictions (Stage 2); the
quality-assurance stage is pure CPU.

### Data

Both datasets are public and are **not** redistributed here.

- **ACDC** — Bernard et al. 2018, [`10.1109/TMI.2018.2837502`](https://doi.org/10.1109/TMI.2018.2837502).
  100 patients, 1 centre, Siemens.
- **M&Ms** — Campello et al. 2021, [`10.1109/TMI.2021.3090082`](https://doi.org/10.1109/TMI.2021.3090082).
  320 patients, 4 manufacturers, 5 centres, 9 pathologies.

### Stage 1 — Preprocessing  (`pipeline/`)

```bash
python pipeline/prep_acdc_4d.py --src <ACDC_root>  --out preprocessed
python pipeline/prep_mnm.py     --src <M&Ms_root>  --out preprocessed_mnm
```

Per short-axis slice: extract the 2D+*t* sequence, clip to the 2nd–98th intensity percentile,
normalise to `[0,1]`, resize to 512 × 512, **harmonise the label convention**, and write one NPZ.

> ⚠️ **The raw M&Ms distribution uses the opposite label convention to ACDC** (`1 = LV, 3 = RV`).
> It is remapped here. Getting this wrong once produced an LV Dice of **0.005** — the code runs
> perfectly and reports nonsense. `src/verify_data.py` therefore re-establishes the convention
> **geometrically** rather than trusting the remap: the LV is the blood pool whose boundary
> neighbourhood is almost entirely myocardium (measured: 1.000 on ACDC, 0.994 on M&Ms for label 3).

### Stage 2 — Segmentation inference  (`pipeline/`)

```bash
python pipeline/infer_medsam2.py \
    --data preprocessed_mnm --out results/medsam2_mnm \
    --ckpt MedSAM2/checkpoints/MedSAM2_latest.pt \
    --cfg  configs/sam2.1_hiera_t512.yaml \
    --overwrite
```

MedSAM2 is prompted with a bounding box at ED and propagated in both temporal directions to cover
every frame; it is prompted again at ES and likewise propagated both ways; the two full-length
predictions are then merged, each frame taking whichever anchor is nearer. **All three predictions are
written**: `ed_pred`, `es_pred`, `bidir` — each `(T, 512, 512)`.

> ⚠️ **`--overwrite` is not optional if any input changed.** Without it the script silently skips
> files that already exist. This once "processed" 2866 files in under a second, reusing stale
> predictions from before a preprocessing fix.
>
> ⚠️ MedSAM2/SAM2 resolve their config through Hydra **relative to the working directory** — the
> `--cfg` path must be relative and the process must `chdir` into `MedSAM2/`. The script does this
> at import.

### Stage 3 — Quality assurance  (`src/`)

**No GPU. No training. No inference.** Reads the NPZs from Stage 2. ≈ 20 min on 48 CPU cores.

```bash
bash jobs/run_all.sh          # runs everything below, in order
bash jobs/watch.sh            # live progress
```

| Script | What it produces | Paper section |
|---|---|---|
| `verify_data.py` | **Blocking gate.** ED/ES indices vs dataset metadata *per manufacturer*; EDV > ESV per patient; label convention re-derived geometrically; the disagreement signal re-checked using **mid-cycle frames only**, where neither pass is anchored. **Exits non-zero on failure.** | §4 (Verification) |
| `compute_metrics.py` | `patient_metrics.csv` — per-patient Dice at the annotated phases and EF/EDV/ESV from predicted *and* reference masks. **The only place ground truth defines the endpoint.** | Fig. 1, Table 2 |
| `extract_features.py` | `patient_features.csv` — 46 no-reference features. **Asserts** that no ground-truth-derived column, and no manufacturer / centre / pathology label, can enter the matrix. Also writes the dense `A(t)` curves. | Table 1 |
| `variant_features.py` | `variant_features.csv` — the same plausibility features computed separately for `bidir`, `ed_pred`, `es_pred`, so they can be *compared*. | Fig. 5 |
| `train_qc.py` | `dice_estimation.csv`, `qc_performance.csv`, `ood_control.csv`, `risk_coverage.csv`, `ablation.csv`, `failure_prevalence.csv`, `qc_scores.csv` — nested CV → **frozen** → external; the appearance-OOD control; risk–coverage; the ablation; threshold sensitivity. | Figs. 3–5, Table 3 |
| `reselect.py` | `reselection.csv` — score all three predictions, keep the most plausible. | Fig. 5, Table 3(d) |
| `make_figures.py` / `make_tables.py` | the 5 figures and 3 LaTeX tables, read from the CSVs | all |
| `verify_numbers.py` | re-derives **every one of the 54 numeric claims** in the manuscript. **Non-zero exit on any mismatch.** | — |

### How EF is computed, and why you can trust it

LV volume at frame *t* is the LV voxel count times the physical voxel volume. Masks are stored at
512 × 512 while `pixdim` is given at the original resolution, so the in-plane spacing is rescaled by
`orig_H/512` and `orig_W/512` first. Then

```
EDV = max(V_ED, V_ES)     ESV = min(V_ED, V_ES)     EF = (EDV − ESV) / EDV × 100
```

`max`/`min` rather than a fixed assignment, because the ED/ES frame *ordering* is not consistent
across manufacturers.

The **reference** EF is computed the same way from the reference masks — which is exactly how EF is
measured clinically from cine CMR (summation of discs). It is checkable, and it checks out:

- 420 / 420 patients have it. **EDV > ESV for 100 % of them.**
- The ACDC pathology groups **recover their known clinical physiology**: **DCM 18.4 %** (dilated
  cardiomyopathy — severely reduced), **MINF 31.0 %** (prior infarction — reduced), **NOR 60.4 %**
  (normal), **HCM 67.4 %** (hypertrophic — preserved). If the EF pipeline were wrong, five pathology
  groups could not each land in their correct clinical band.

---

## Repository layout

```
pipeline/   prep_acdc_4d.py     raw 4D cine + ED/ES masks -> 512x512 NPZ; label convention harmonised
            prep_mnm.py         the same, for the multi-vendor cohort
            infer_medsam2.py    dual-anchored propagation -> ed_pred / es_pred / bidir
            infer_sam2.py       the non-medical SAM2 baseline

src/        common.py           shared loaders, geometry, metric primitives
            verify_data.py      the blocking gate (see Stage 3)
            compute_metrics.py  reference metrics -- the only place GT defines the endpoint
            extract_features.py 46 no-reference features + a leakage assertion
            variant_features.py per-variant features, for reselection
            train_qc.py         nested CV -> frozen -> external; OOD control; risk-coverage; ablation
            reselect.py         QC-guided reselection
            make_figures.py     5 figures      make_tables.py   3 LaTeX tables
            verify_numbers.py   every manuscript number, re-derived

docs/       PIPELINE.md         end to end, including how EF is computed and why it is trustworthy
            DATASETS.md         cohorts, label convention, and the verification gate
            MODELS.md           what each prediction directory is, and what is excluded and why
            SKILLS.md           leakage control, nested CV, calibration, selective prediction -- and why
            ENVIRONMENT.md      cluster, conda, and the traps that cost real time
            RESULTS_SUMMARY.md  every headline number in one place

configs/    cycleqa.yaml        thresholds, CV settings, seeds -- one source of truth
jobs/       run_all.sh          regenerates every CSV, figure and table from the raw NPZs
results/    *.csv               all results; every manuscript number traces to one of these
figures/    fig1..fig5.png
tables/     table1..table3.tex
metispaper.md                   section-by-section blueprint: what each figure and table must argue
```

*(The manuscript source is not included while the paper is in press.)*

---

## Honesty notes

These are in the paper. They belong here too.

- **Prompts are derived from the reference masks.** Every number is an **oracle-prompt upper bound**;
  a deployed system additionally needs automatic prompt generation. This makes the finding
  *stronger*: even with perfect prompts, the model breaks on an unseen manufacturer.
- **Reference masks exist only at ED and ES.** We validate annotated-phase error and claim nothing
  about intermediate phases. Full-cycle features are *inputs*, never validated targets.
- **Learning is not what makes this work.** Anchor features fitted with logistic regression score
  0.987 internally but **0.793** externally — textbook overfitting on 100 patients containing 11
  failures. The **untrained** rule transfers *better* (**0.850**). Reported rather than hidden: the
  disagreement signal is strong enough that no training is required, and on a development set this
  small, training on it is actively harmful.
- **Retrospective, public data. No prospective clinical validation.**
- Supervised U-Net and DINOv2 predictions exist for the external cohort, but their checkpoint
  provenance could not be established, so they are excluded rather than defended
  ([`docs/MODELS.md`](docs/MODELS.md)).

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
