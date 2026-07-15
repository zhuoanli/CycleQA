# The paper, explained section by section

This is the blueprint for `paper/main.tex`. For every section it says **what goes in it, what
it must argue, and which figure or table carries the argument**. Read this before touching the
LaTeX.

---

## 0. First, the vocabulary

You need these to read anything below.

| Term | What it means |
|---|---|
| **Dice** | Overlap between the predicted mask and the true mask, 0–1. 0.9 = excellent, 0.7 = marginal, 0.5 = useless. It is what segmentation papers report. |
| **EF** — ejection fraction | `(EDV − ESV) / EDV × 100`. The fraction of blood the left ventricle pumps out per beat. **This is the number a cardiologist actually reads.** Normal ≈ 55–70%. Below 40% = heart failure. |
| **EDV / ESV** | End-diastolic volume (heart full) and end-systolic volume (heart emptied), in mL. |
| **pp** — percentage point | EF is itself a percentage, so "error = 6.4 **pp**" means the estimate was 6.4 *points* off (e.g. said 66%, truth 60%). Writing "6.4%" would be ambiguous — it would sound like a *relative* error. **An EF error of 23 pp can turn a healthy heart (60%) into reported heart failure (37%).** That is why the unit matters. |
| **AUROC** | How well a detector separates two groups. 0.5 = coin flip. 1.0 = perfect. 0.85 = strong. |
| **AUPRC** | Like AUROC but better behaved when one class is rare. |
| **MAE** — mean absolute error | Average size of the mistake. "Dice estimated to MAE 0.072" = on average our guess is 0.072 off the truth. |
| **Brier score** | **Calibration.** AUROC only checks *ranking*. Brier checks whether "80% risk" studies actually fail 80% of the time. A triage system must be calibrated or the risk number is meaningless. Lower is better. |
| **Out-of-fold** | A prediction made by a model that never saw that patient during training. The only honest way to report performance on the development set. |
| **Frozen** | Fitted once, then never touched again. Our risk model is frozen after development and applied to the external cohort exactly once. |

### Is the EF ground truth real? Yes — and here is the proof.

Every one of the 420 patients has a **ground-truth EF**, computed from the *reference masks* at
end-diastole and end-systole. This is exactly how EF is measured clinically from cine CMR
(summation of discs over the short-axis stack). Checks:

- 420/420 patients have it. **Zero missing.**
- **EDV > ESV for 100% of patients** — physically necessary; if it failed, the frames would be swapped.
- The ACDC pathology groups **recover their known clinical physiology**:

| Group | Ground-truth EF | Clinically expected |
|---|---|---|
| **DCM** (dilated cardiomyopathy) | **18.4%** | severely reduced ✓ |
| **MINF** (prior infarction) | **31.0%** | reduced ✓ |
| **RV** (right-ventricular abnormality) | 54.7% | LV preserved ✓ |
| **NOR** (normal) | **60.4%** | 55–65% ✓ |
| **HCM** (hypertrophic) | **67.4%** | preserved / high ✓ |

If the EF pipeline were wrong, five pathology groups could not each land in their correct
clinical band. This is the validation.

---

## 1. Title

> **Cross-Vendor Quality Assurance of Zero-Shot Full-Cycle 4D Cardiac Cine MRI Segmentation
> via Dual-Anchored Medical Video Foundation Model**

**Author: Zhuoan Li** (sole author), Weldon School of Biomedical Engineering, Purdue University.
ORCID 0009-0005-0647-8579.


---

## 2. Abstract (~250 words)

Must contain, in this order:

1. **The promise.** Video foundation models segment the whole cardiac cycle zero-shot from two
   box prompts. Dice 0.913/0.801/0.858 on the public benchmark. No training.
2. **The trap.** A model that needs no labels to *run* has no labels with which to be *checked*.
3. **The finding.** On 320 external patients across 4 manufacturers: Dice falls modestly, but
   **EF error rises 6.4 → 23.4 pp** and failed studies rise **11% → 54%**. The degradation
   tracks the *manufacturer*, not the dataset.
4. **The idea.** Dual anchoring propagates the same video twice. If the model is right the two
   passes converge; if it drifts they separate. **The disagreement is the error.**
5. **The numbers.** Zero training: Dice estimated to **MAE 0.072** (r = 0.814) externally.
   Failure detection **AUROC 0.850**, including **0.806 on Canon**, a manufacturer never seen.
   An appearance-based detector gets 0.573.
6. **The action.** Reselecting among predictions already on disk: **EF error 23.4 → 12.0 pp**,
   failures **54% → 15%**. No labels, no retraining, no extra inference.
7. **The line.** *The redundancy introduced to make the model accurate also makes it
   self-auditing.*

---

## 3. Introduction (~1.5 pages)

**Paragraph 1 — the promise.** Promptable video foundation models (SAM2, MedSAM2) made zero-shot
segmentation of dynamic medical imaging practical. In cine CMR, a **dual-anchored propagation
scheme** — described in §Methods, *not cited* — segments the entire cardiac cycle from two
bounding boxes with no cardiac-specific training. Its whole appeal for clinical translation is
that it needs no site-specific labels.

**Paragraph 2 — the trap. This is the paper's hook.**
> A model that needs no labels to *run* also has no labels with which to be *checked*.

Benchmark accuracy is measured where annotations exist. Deployment happens where they do not.
And the benchmark is narrow: ACDC is one hospital, one manufacturer. Cite
`varoquaux2022failures` (why medical-imaging ML fails to translate) and `moor2023gmai`.

**Paragraph 3 — the two questions.**
1. Does benchmark accuracy survive a change of scanner? → answered on M&Ms (320 patients,
   4 manufacturers, 5 centres, 9 pathologies), never previously reported for this system.
2. **At a site with no ground truth, can the failure be *seen*?** → the contribution.

**Paragraph 4 — the mechanism, stated once, plainly.** Dual anchoring produces **two
statistically independent propagations of the same video**: one anchored at ED, one at ES, each
propagated in both temporal directions so both span every frame, both covering every frame, both already written to disk. If the model tracks
the true anatomy they must converge. If it drifts they separate. *The magnitude of the
disagreement is the magnitude of the error* — and it needs no reference mask. We call this
**CycleQA**, after the cardiac cycle that must close.

**Paragraph 5 — contributions**, as a numbered list of exactly three:
1. **External validation.** Dice degrades modestly off-vendor; the clinical biomarker degrades
   **sixfold**. Benchmark Dice *understates* the clinical damage.
2. **CycleQA.** Zero-training disagreement estimates Dice to MAE 0.072 and detects failures at
   AUROC 0.850 on unseen manufacturers. A control experiment shows it detects **the error, not
   the scanner**.
3. **Acting on it.** Label-free reselection halves EF error (23.4 → 12.0 pp).

---

## 4. Related Work (~1 page)

Three subsections. **Be honest in the third — this is where a reviewer will attack.**

**4.1 Promptable foundation models for CMR.** SAM → SAM2 → MedSAM/MedSAM2. Prompting studies for
cardiac MRI. CineMA. Cite: `kirillov2023sam`, `ravi2024sam2`, `ma2024medsam`, `zhu2024medsam2`,
`stein2023samcardiac`, `chen2025cinecmrsam`, `fu2025cinema`.

**4.2 Domain shift in cardiac MRI.** Multi-vendor generalisation is a known weakness of
*supervised* CMR segmentation — that is why the M&Ms challenge exists. Cite `tao2019multivendor`,
`chen2020generalizability`, `campello2021mnms`, `martinisla2023mnm2`, `puyolanton2021fairness`.
**Then the twist:** zero-shot models are widely *assumed* to escape this, having never been
trained on the task. They do not. Their **pretraining** corpus carries a manufacturer
distribution, and we show the resulting bias is severe.

**4.3 Segmentation quality control without ground truth. — say plainly that we did not invent
this.**

| Prior method | What it needs | Why it cannot do what we do |
|---|---|---|
| Reverse Classification Accuracy (`valindria2017rca`, `robinson2017qc`) | an atlas database + a **registration per case** | slow; registration is itself appearance-dependent |
| CNN quality regression (`robinson2019qc`, `tarroni2019qc`, `tarroni2020qc`) | a large **labelled** training set; **reads the image** | learns source-manufacturer appearance → no reason to transfer |
| Generative shape prior (`wang2020genqc`) | train a generative model | closest relative; still needs training |
| Ensemble disagreement (`hann2021qc`) | **N models, N forward passes** | ensembles agree *confidently and wrongly* under shift |
| Model uncertainty (`kendall2017uncertainties`, `roy2019`) | model internals, many passes | shown unreliable per case (`jungo2019`) |
| Appearance-based OOD (`hendrycks2017baseline`, `lee2018mahalanobis`, `gonzalez2022ood`) | image statistics | detects that the **input** is unusual ≠ that the **output** is wrong |

**Then state the three gaps we fill:**
1. Every method above targets a **supervised model trained on the deployment domain**. None
   targets a zero-shot **video** foundation model, whose dominant failure mode — temporal drift
   through a causal memory bank — **does not exist for a frame-wise CNN**.
2. Every method above is developed and evaluated **within one domain**. None has been frozen on
   one manufacturer and validated on another. **That is the only question deployment asks.**
3. CycleQA's "ensemble" is **free**: it is the two propagation directions the method already
   computes. No atlas, no registration, no second network, no extra forward pass.

---

## 5. Methods (~2.5 pages)

**5.1 The system under audit.** Describe dual-anchored propagation **in prose — do not cite it as
prior work**. MedSAM2 (hiera-tiny) is prompted with a bounding box at ED and propagated in both temporal
directions; prompted again at ES and likewise propagated both ways; the two full-length predictions
are merged, each frame taking the nearer anchor.
This writes **three full-length predictions per slice**: `M^ED`, `M^ES`, and the merged `M^bi`.
It is treated as **fixed** and is not modified.

> **State plainly:** the prompts are derived from the reference masks, so **every number in this
> paper is an oracle-prompt upper bound**. A deployed system additionally needs automatic prompt
> generation. Saying this makes the finding *stronger*: even with perfect prompts, it breaks
> off-vendor.

**5.2 CycleQA features → Table 1.** Four families, 46 features, all computed from predicted masks
alone. The primary signal:

> For frame *t* and structure *c*:  **A_c(t) = Dice( M^ED_c(t), M^ES_c(t) )**

The two propagations traverse the memory bank in opposite directions and accumulate *independent*
error. Agreement is evidence both are tracking the same — and plausibly the true — anatomy.
Disagreement is unambiguous evidence at least one has drifted.

The other three families encode constraints that are true **on any scanner by construction**: the
myocardium must form a closed ring around the LV cavity; V(t) must be smooth and close its cycle;
EDV must exceed ESV.

> **The leakage rule, stated explicitly:** ground truth, vendor, centre and pathology may **never**
> enter the feature matrix. This is enforced by an assertion in the code.

**5.3 Endpoints.** Primary: **any structure with Dice < 0.70** at an annotated phase. Secondary:
**|ΔEF| > 10 pp**. Both pre-specified and **swept** in a sensitivity analysis; **neither was chosen
to maximise AUROC**. Reference masks exist only at ED/ES, so the endpoint is *annotated-phase*
error — full-cycle features are **inputs**, never validated targets. **Make no claim about
intermediate phases.**

**5.4 Protocol — this paragraph is what makes the paper credible.**
- **Tier 1, training-free:** mean disagreement used *directly* as an estimate of Dice. Nothing is
  fitted.
- **Tier 2, calibrated:** a risk model (rule-based / **logistic regression** / random forest) is
  fitted on the 100 development patients with patient-level **nested 5-fold cross-validation** —
  imputation, scaling and hyperparameter search happen strictly *inside* each training fold, and
  only out-of-fold predictions are reported. Then **frozen** and applied **once** to the 320
  external patients. It never sees a Philips, GE or Canon study during development.
- Confidence intervals: **patient-level** bootstrap, 2000 resamples. (Slice-level resampling would
  give CIs that are far too narrow — slices within a patient are not independent.)

> **What "training the logistic regression" actually means, since you asked:** you have 46 numbers
> per patient (the features) and one label per patient (did it fail?). Logistic regression learns
> a weighted sum of those 46 numbers that best separates failures from non-failures. On the 100
> development patients only. Then the weights are **frozen** and used, unchanged, on the 320
> external patients. The point is not that logistic regression is clever — it is that we can check
> whether what it learned on Siemens still holds on Canon.

**5.5 The control — what image-OOD is for.** A reviewer will ask: *"Isn't CycleQA just a Canon
detector?"* If it merely recognises unusual-looking images it is useless — it cannot separate
"unusual-looking but correct" from "broken". So we build an **appearance-only** detector (per-patient
intensity moments, scored by Mahalanobis distance to the development mean) and score **every**
detector against **two different questions**:

- **Q1: is this an unusual scanner?**
- **Q2: is the segmentation wrong?**

A vendor classifier is good at Q1 and bad at Q2. An error detector is the reverse. That is the
experiment.

**5.6 Interventions.** (a) **Selective referral**: rank by risk, send the top X% for human review,
report the residual error among the auto-reported rest. (b) **CycleQA-guided reselection**: score
each of the three predictions already on disk and keep the most plausible.

---

## 6. Results (~3.5 pages)

### 6.1 "It is the manufacturer, not the dataset."  →  **Figure 1** + **Table 2**

**What Figure 1 says, and why it is built the way it is.**

The naive comparison — *development vs external* — is **confounded**: the development cohort is
acquired on Siemens scanners, so moving to the external cohort changes the dataset, the hospital
*and* the manufacturer all at once. Figure 1 breaks that apart with a **three-way split**:

| Group | What changed | EF error |
|---|---|---|
| **ACDC (Siemens)** | — (development) | **6.4 pp** |
| **M&Ms Siemens** | new dataset, new hospital, **same manufacturer** | **5.7 pp** ← *nothing happened* |
| **M&Ms Philips / GE / Canon** | **new manufacturer** | **29.0 / 32.5 / 33.9 pp** ← *collapse* |

> **A new hospital on the same scanner costs nothing. A new scanner costs everything.**
> This is not "external data is hard." It is *specifically* the manufacturer.

Three panels, deliberately:
- **(a) Dice** — degrades *modestly* (0.857 → 0.705–0.746). A reader looking only at Dice would
  conclude the system is degraded but usable.
- **(b) EF error** — degrades **sixfold**. The masks that produced Dice 0.75 produce a *clinically
  worthless* biomarker.
- **(c) Failure rate** — 11% → 70%.

**The point of putting (a) next to (b): benchmark Dice conceals the clinical damage.** That single
juxtaposition is the paper's motivation.

### 6.2 "The failure is visible with no ground truth."  →  **Figure 2** + **Table 3(a)**

**What Figure 2 says.** A scatter: **x = true Dice (needs a reference mask). y = CycleQA's estimate
(needs nothing).** If the points lie on the diagonal, you can measure segmentation quality at a
hospital that has no annotations.

They do. **No model is fitted anywhere in this figure.** It is the raw disagreement.

| | Pearson r | MAE |
|---|---|---|
| ACDC (development) | 0.735 | 0.064 |
| **M&Ms (external)** | **0.814** | **0.072** |

> **The signal is *stronger* out of domain than in it** — it is most informative exactly where it
> is most needed, because the errors there are larger and more varied.

### 6.3 "It transfers to scanners it has never seen."  →  **Figure 3** + **Table 3(b)**

**What each panel of Figure 3 says:**

- **(a) Detecting failed studies.** ROC curves on the external cohort, **one per manufacturer**.
  The model was frozen after development and has **never seen** Philips, GE or Canon.
  AUROC: Siemens 0.883 · Philips 0.857 · GE 0.823 · **Canon 0.806**. All well above 0.8.
  **This panel is the paper's central claim.**

- **(b) What is actually detected? — the control.** Two detectors × two questions:

  | | *Q1: is it an unusual scanner?* | *Q2: is the segmentation wrong?* |
  |---|---|---|
  | **Looks at the IMAGE** (appearance OOD) | 0.48 | 0.57 |
  | **Looks at the MASK** (CycleQA) | 0.64 | **0.85** |

  CycleQA is **far better at finding the error (0.85) than at finding the scanner (0.64)**. It is
  an *error detector*, not a vendor classifier. The reviewer's objection is answered.

  **Report honestly:** the appearance-based detector is near chance at *both*. Standard
  preprocessing (percentile clipping + intensity normalisation) has already removed most of the
  raw appearance difference — **yet the segmentation still fails**. The conclusion is not a neat
  double dissociation; it is stronger and simpler: **appearance-based OOD detection is not a
  viable safety net here.** The shift is not one that intensity statistics can see.

- **(c) Is the risk trustworthy? — calibration.** AUROC only checks *ranking*. This panel checks
  whether a reported risk of 0.8 corresponds to an actual failure rate of 0.8. It roughly does
  (Brier = 0.204). **Without this, "high risk" would be a label with no meaning, and a triage
  system built on it would be unsafe.**

### 6.4 The ablation carries an honest surprise.  →  **Table 3(c)**

| Features | Internal (out-of-fold) | **External** |
|---|---|---|
| Anchor disagreement, **fitted** with logistic regression | 0.987 | **0.793** |
| Plausibility features only (no anchor) | 0.863 | **0.891** |
| **Anchor disagreement, NOT fitted (raw rule)** | 0.909 | **0.850** |

An internal 0.987 against an external 0.793 is a **textbook overfitting signature** — unsurprising
when 24 features are fitted to 100 patients containing 11 failures.

> **Do not hide this. State it as the finding it is:**
> **We do not claim that learning is what makes CycleQA work. The opposite is true, and it is more
> useful: the disagreement signal is strong enough that no training is required — and on a
> development set this small, training on it is actively harmful.**

### 6.5 "Acting on the risk score."  →  **Figure 4** + **Table 3(d)**

**Definitions you asked for, stated in the paper:**

- **"Today" / current system** = what the pipeline does now: always report the **merged**
  (`bidir`) output. This is the baseline every number is compared against.
- **"Best fixed (needs labels)"** = always use the ES-anchored propagation. It happens to be the
  best *single* choice on the external cohort — but **you could only know that by having the
  labels you do not have**. It is a hindsight baseline, not an available strategy.
- **"Oracle (upper bound)"** = if you had ground truth and could pick the best of the three
  predictions *for each patient individually*. This is the ceiling, not a method.
- **"CycleQA"** = score each of the three predictions with the no-reference risk model and keep
  the most plausible. **No labels. No retraining. No extra inference — all three predictions are
  already on disk.**

**Panels:**
- **(a) Selective referral.** Rank by risk, send the riskiest for human review. Reviewing the
  riskiest 20% drops residual EF error 23.4 → 19.8 pp. **Real but modest** — with a 54% failure
  rate, a 20% review budget cannot fix much. Say so.
- **(b) Reselection.** **23.4 → 12.0 pp.** Beats the best fixed choice (16.8), approaches the
  oracle (10.4).
- **(c) Per manufacturer — this panel reveals the mechanism.** On Siemens, CycleQA keeps the merged
  or ED-anchored output and preserves EF error at 5.8 pp — whereas *always* choosing ES, the best
  global strategy, would have **wrecked** it (14.8 pp). On Canon/GE/Philips it switches to ES.

> **No fixed strategy works everywhere. CycleQA adapts per patient and per manufacturer — without
> ever being told which manufacturer it is looking at.**

---

## 7. Discussion (~1.5 pages)

**7.1 Why quality assurance transfers when segmentation does not. — the intellectual core.**

The segmenter must answer a **hard** question — *which pixels are myocardium?* — and to do that it
must read **appearance**. Appearance is exactly what changed between manufacturers. So it broke.

CycleQA asks two **easy** questions — *are you consistent with yourself?* and *is your answer
anatomically possible?* Neither requires the image. **A heart is a heart on any scanner:** the
myocardium encloses the LV cavity, the volume curve closes, EDV exceeds ESV. These constraints are
vendor-invariant **by construction**, which is why a detector built on them survives a change of
manufacturer that the segmenter does not.

> **Verifying an answer is easier than producing it. That is why the watchdog is more robust than
> the model it watches.**

**7.2 Deployment and post-market surveillance.** A site installs the tool with no labels and no
retraining. Every study gets a CycleQA score in ~1 second of CPU — no second network. Low risk →
auto-report. High risk → flagged; the risk–coverage curve turns review capacity into a **budgeted**
decision. Where a more plausible alternative propagation exists → substituted automatically.

**And the part that matters most for regulation:** the **mean** CycleQA risk across a site's
workload is itself a **label-free proxy for performance**. A drift above the development baseline is
an alarm that the site is out of distribution — **raised before any patient has been misreported**.
Regulators require post-market surveillance of deployed AI devices, and the usual obstacle is that
the real world supplies no ground truth. **CycleQA supplies a surrogate.**

**7.3 Limitations — state all of these.**
- Prompts are reference-derived → **oracle-prompt upper bound**. Deployment needs automatic prompting.
- Reference masks exist **only at ED/ES** → annotated-phase error only; no claim about intermediate phases.
- Retrospective, public data. **No prospective clinical validation.**
- The review threshold is an operating choice each site must set against its own risk tolerance.
- The development set is small (100 patients, 11 failures) — **which is exactly why the
  training-free tier, not the learned one, is the recommended configuration.**

---

## 8. Conclusion (~0.3 pages)

A zero-shot cardiac video foundation model that scores 0.91 Dice on a public benchmark produces a
34-point ejection-fraction error on an unseen scanner — and at that site there is **no ground truth
with which to notice**. The failure is nonetheless legible from the predicted masks alone, because
dual-anchored propagation runs the same video twice and the two passes disagree **exactly when the
model is wrong**. Used without any training, that disagreement estimates Dice to within 0.072 and
detects failed studies at AUROC 0.850 on manufacturers absent from development; used to reselect
among predictions already computed, it halves ejection-fraction error.

> **The redundancy introduced to make the model accurate is the redundancy that makes it
> self-auditing. A fragile model can be deployed safely if the watchdog watching it is not fragile.**

---

## 9. Figures — placement and purpose

There are **six**. Figure 1 is the graphical overview (the uploaded `method.png`); Figure 3 is the qualitative one: a paper about segmentation quality that never
shows a segmentation is not persuasive, and this figure lets the reader *see* the method work
before a single statistic is quoted.

| Figure | File | Goes in | **The one sentence it must make the reader believe** |
|---|---|---|---|
| **1** | `method.png` | §5 (Methods opening) | *The whole method in one picture: two propagations → disagreement → risk → act, all without labels.* |
| **2** | `fig1_vendor.png` | §6.1 | *It is the manufacturer, not the dataset — and benchmark Dice conceals the clinical damage.* |
| **3** | `fig2_qualitative.png` | §6.2 (open with it) | *You can literally see the two propagations contradicting each other exactly where the segmentation fails — and no reference mask was used to see it.* |
| **4** | `fig3_estimation.png` | §6.2 | *Segmentation quality is estimable from the predictions alone, with no training at all.* |
| **5** | `fig4_detection.png` | §6.3 | *The detector transfers to scanners it has never seen, and it is detecting the error, not the scanner.* |
| **6** | `fig5_action.png` | §6.5 | *The risk score is actionable: it halves the clinical error, for free.* |

### Figure 2 in detail — what the reader is looking at

Two patients, five columns. Both rows show a **mid-cycle frame**, deliberately: that is where
*neither* propagation is anchored, so neither has the advantage of a prompt.

**Left of the divider — the annotated phase (ED), where a reference mask exists:**

| Column | What it shows |
|---|---|
| 1 | **Reference** — the ground-truth mask |
| 2 | **Prediction** — what the model produced. The reader sees the *true* error here. |

**Right of the divider — mid-cycle, where no annotation exists at all:**

| Column | What it shows |
|---|---|
| 3 | **ED-anchored** — the ED-anchored prediction at this frame (it reached here by propagating from ED) |
| 4 | **ES-anchored** — the ES-anchored prediction at this frame (it reached here by propagating from ES) |
| 4 | **the disagreement** — computed as `ed_pred[t] != es_pred[t]`: the voxels where the two passes assign *different labels*. **This is the only thing CycleQA is allowed to see.** |
| 5 | **the disagreement map** |
| 6 | agreement `A(t)` across the cardiac cycle, and CycleQA's risk |

> **How the disagreement is computed — no ground truth anywhere in it:**
> for frame *t* and structure *c*, `A_c(t) = Dice( ed_pred[t] , es_pred[t] )` — the overlap between
> the two independent propagations. The map in column 5 is simply `ed_pred[t] != es_pred[t]`.

Both cases are displayed in **standard radiological short-axis orientation** (right ventricle to the left of the left ventricle, septum vertical); this is a per-case rotation for display only and enters no computation. Both rows show a **mid-ventricular** level. Basal and apical slices are deliberately excluded: the
right ventricle is truncated at the base and the cavity vanishes at the apex, so neither shows the
whole heart. The crop is tight to the bounding box of the reference and both propagations.

**Top row (Siemens, PASS):** all three structures are segmented well (RV 0.918, Myo 0.813, LV
0.886). Reference and prediction are nearly identical, and so are the two propagations. The
disagreement map is a thin rim — boundary noise. **CycleQA risk 0.12. True Dice 0.872. EF error
0.2 pp.** (A risk of exactly 0.00 would look artificial; 0.12 is a real, low, non-degenerate score.)

**Bottom row (Canon, REVIEW):** the two propagations place the myocardium and cavity differently, and
the disagreement map becomes a **large solid region covering the entire left ventricle** — the model
is visibly contradicting itself. Myocardial agreement collapses to ≈0.35. **CycleQA risk 0.99. True
Dice 0.620. EF error 41.7 pp.**

> **Neither judgement used a reference mask. The red region *is* the signal.** This is the figure
> that makes the method intuitive before any number is quoted.

All five figures are new. All are print-safe in greyscale — manufacturer identity is carried by
**marker shape and hatch as well as colour**, never colour alone.

## 10. Tables — placement and purpose

| Table | File | Goes in | **What it must establish** |
|---|---|---|---|
| **1** | `table1_features.tex` | §5.2 | The 46 features, and — critically — **why each family is vendor-invariant by construction**. This table is the argument for why the method transfers, not just a list. |
| **2** | `table2_external.tex` | §6.1 | The external-validation numbers behind Figure 1, per manufacturer, with the oracle-prompt caveat in the caption. |
| **3** | `table3_qc.tex` | §6.2–6.5 | Four blocks: **(a)** training-free Dice estimation; **(b)** failure detection, internal out-of-fold vs external, per manufacturer, with bootstrap CIs; **(c)** the ablation (including the overfitting result); **(d)** acting on the risk score. |

---

## 11. References

`paper/references.bib` — **every new entry was checked against a live Crossref DOI or arXiv ID
before being added. Do not add an unverified entry.**

| Group | Keys |
|---|---|
| Datasets | `bernard2018acdc`, `campello2021mnms`, `martinisla2023mnm2` |
| Foundation models | `kirillov2023sam`, `ravi2024sam2`, `ma2024medsam`, `zhu2024medsam2`, `stein2023samcardiac`, `chen2025cinecmrsam`, `fu2025cinema` |
| **QC without ground truth** (the direct prior art — **must be positioned against**) | `valindria2017rca`, `robinson2019qc`, `robinson2017qc`, `tarroni2019qc`, `tarroni2020qc`, `wang2020genqc`, **`hann2021qc`**, `kohlberger2012`, `ruijsink2020` |
| CMR foundation-model applications | `shahin2026adapting`, `li2026classification` |
| CMR quality control / uncertainty (group) | `yalcinkaya2024qc`, `yalcinkaya2024uncertainty` |
| OOD / uncertainty | `gonzalez2022ood`, `jungo2019`, `roy2019`, `hendrycks2017baseline`, `lee2018mahalanobis`, `kendall2017uncertainties`, `lakshminarayanan2017ensembles` |
| Calibration / selective prediction | `guo2017calibration`, `geifman2017selective` |
| Domain shift / translation | `tao2019multivendor`, `chen2020generalizability`, `puyolanton2021fairness`, `varoquaux2022failures`, `moor2023gmai` |

**Self-citations are written in the third person** ("Yalcinkaya et al. showed..."), never "our previous work", to preserve double-blind anonymity. The bib contains **only entries actually cited** in main.tex (35 total; no redundant entries).

**One entry was deliberately dropped:** El-Yaniv & Wiener 2010 (JMLR) could not be verified on
Crossref, DBLP or arXiv. `geifman2017selective` covers selective classification.
**Never cite what has not been checked.**

**`hann2021qc` is the closest competing method. Omitting it would be fatal — a reviewer will know it.**

---

## 12. Before submitting

```bash
python src/verify_numbers.py     # all 54 numeric claims re-derived from CSV; exits non-zero on mismatch
```

Then: compile on Overleaf (no LaTeX on the cluster login node), sign the Springer Licence to
Publish **with this title**, build the source ZIP, upload.
