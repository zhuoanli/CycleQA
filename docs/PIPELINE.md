# The full pipeline, end to end

This repository is self-contained: it covers raw DICOM/NIfTI → preprocessing → segmentation
inference → quality assurance → figures. Nothing is assumed to exist beforehand except the two
public datasets and the MedSAM2 checkpoint.

```
raw NIfTI                 preprocessing            inference              quality assurance
─────────                 ─────────────            ─────────              ─────────────────
ACDC  ──┐                 prep_acdc_4d.py  ──┐     infer_medsam2.py  ──┐   verify_data.py
        ├─ 4D cine + ──►                     ├──►  (dual-anchored)     ├─► compute_metrics.py
M&Ms  ──┘  ED/ES masks    prep_mnm.py     ──┘      ed_pred/es_pred/    │   extract_features.py
                                                    bidir              │   train_qc.py
                                                                       └─► reselect.py
```

---

## Stage 1 — Preprocessing  (`pipeline/prep_acdc_4d.py`, `pipeline/prep_mnm.py`)

**Input.** The raw 4D cine volume (`X, Y, Z, T`) and the reference masks at ED and ES.

**What it does, per short-axis slice:**

1. Extract the slice as a 2D+*t* sequence: `(T, H, W)`.
2. Clip intensities to the 2nd–98th percentile (removes outlier voxels).
3. Normalise to `[0, 1]`.
4. Resize to `512 × 512` (MedSAM2's input size), replicated to 3 channels.
5. **Harmonise the label convention** (see below).
6. Write one NPZ per slice.

**Output NPZ keys:**

| key | shape | meaning |
|---|---|---|
| `frames` | `(T, 3, 512, 512)` float16 | the cine sequence |
| `ed_mask`, `es_mask` | `(512, 512)` uint8 | reference masks — **these exist at ED and ES only** |
| `ed_idx`, `es_idx` | scalar | which frame is ED / ES |
| `pixdim` | `(3,)` | voxel spacing in mm, at the **original** resolution |
| `orig_H`, `orig_W` | scalar | pre-resize dimensions (needed to convert voxel counts to mL) |
| `group` | scalar | pathology label |

### The label convention — verify it, never assume it

**Both preprocessed caches use `1 = RV, 2 = Myocardium, 3 = LV`.**

The raw M&Ms distribution ships the **opposite** convention (`1 = LV, 3 = RV`). It is remapped
during preprocessing. Getting this wrong once produced an LV Dice of **0.005** — the failure is
silent and catastrophic, because the code runs perfectly and reports nonsense.

Rather than trust the remap, `src/verify_data.py` re-establishes it **geometrically**: the LV is
the blood pool whose boundary neighbourhood is almost entirely myocardium. Measured:

| | label 1 myo-enclosure | label 3 myo-enclosure | ⇒ LV is |
|---|---|---|---|
| ACDC | 0.335 | **1.000** | label 3 |
| M&Ms | 0.258 | **0.994** | label 3 |

---

## Stage 2 — Segmentation inference  (`pipeline/infer_medsam2.py`)

**Dual-anchored propagation.** MedSAM2 (hiera-tiny) is a *promptable video* model: prompt it on
one frame and it propagates the mask through the sequence via a causal memory bank. Error
accumulates with distance from the prompt.

The scheme prompts it **twice**:

| prediction | prompted at | propagates | written as |
|---|---|---|---|
| ED-anchored | **ED** frame | both directions, spanning every frame | `ed_pred` |
| ES-anchored | **ES** frame | both directions, spanning every frame | `es_pred` |
| merged | — | each frame taken from its nearest anchor, split at the temporal midpoint | `bidir` |

Prompts are **bounding boxes** derived from the reference mask at ED/ES, padded 10 px, one per
structure.

> **Prompts come from the reference masks**, so every number reported is an **oracle-prompt upper
> bound**. A deployed system additionally needs automatic prompt generation.

All three predictions are written to disk. `ed_pred` and `es_pred` are the basis of the
quality-assurance method: they are two statistically independent traversals of the same video, and
their disagreement is the signal.

```bash
python pipeline/infer_medsam2.py \
    --data preprocessed_mnm --out results/medsam2_mnm \
    --ckpt MedSAM2/checkpoints/MedSAM2_latest.pt \
    --cfg  configs/sam2.1_hiera_t512.yaml \
    --overwrite                      # <-- see below
```

**Two notes:**
- **`--overwrite` is not optional if any input changed.** Without it the script silently skips
  files that already exist. This once "processed" 2866 files in under a second, reusing stale
  predictions from before a preprocessing fix.
- MedSAM2/SAM2 resolve their config through Hydra **relative to the working directory**. The
  `--cfg` path must be relative, and the process must `chdir` into `MedSAM2/`. The script does
  this at import.

**Output NPZ keys:** `ed_pred`, `es_pred`, `bidir` — each `(T, 512, 512)` uint8 — plus `ed_idx`,
`es_idx`, `pixdim`, `group`, `orig_H`, `orig_W`.

---

## Stage 3 — Quality assurance  (`src/`, this paper)

**No GPU. No training. No inference.** Everything below reads the NPZs from Stage 2.

| Script | What it does |
|---|---|
| `verify_data.py` | **Blocking gate.** Verifies ED/ES indices against dataset metadata *per manufacturer*, checks EDV > ESV per patient, re-establishes the label convention geometrically, and confirms the disagreement signal survives restriction to mid-cycle frames. **Exits non-zero on failure.** Run this first, always. |
| `compute_metrics.py` | Per-patient reference metrics: Dice at the annotated phases, and EF/EDV/ESV from both predicted and reference masks. **This is the only place ground truth defines the endpoint.** |
| `extract_features.py` | 46 no-reference features per patient. **Asserts** that no ground-truth-derived column, and no manufacturer / centre / pathology label, can enter the matrix. |
| `variant_features.py` | The same plausibility features, computed separately for `bidir`, `ed_pred` and `es_pred` — needed to *choose between* them. |
| `train_qc.py` | Nested cross-validation on the development set → **frozen** → applied once to the external set. Plus the appearance-OOD control, risk–coverage, ablation and threshold sensitivity. |
| `reselect.py` | Score all three predictions with the frozen model and keep the most plausible. |
| `make_figures.py`, `make_tables.py` | All five figures and three tables, read from the CSVs. |
| `verify_numbers.py` | Re-derives **every numeric claim in the manuscript** from the CSVs. Exits non-zero on any mismatch. |

### How EF is computed, and why you can trust it

From the mask stack, LV volume at frame *t* is the LV voxel count times the physical voxel volume.
Because masks are stored at 512×512 but `pixdim` is given at the original resolution, the in-plane
spacing is rescaled by `orig_H/512` and `orig_W/512` before multiplying. Then

```
EDV = max(V_ED, V_ES)      ESV = min(V_ED, V_ES)      EF = (EDV − ESV) / EDV × 100
```

`max`/`min` rather than a fixed assignment, because the ED/ES frame *ordering* is not consistent
across manufacturers.

**Reference EF is computed the same way from the reference masks** — which is exactly how EF is
measured clinically from cine CMR (summation of discs). It is real, and it is checkable:

- 420/420 patients have it. **EDV > ESV for 100% of them.**
- The pathology groups recover their known clinical physiology: **DCM 18.4%** (severely reduced),
  **MINF 31.0%** (reduced), **NOR 60.4%** (normal), **HCM 67.4%** (preserved). If the EF pipeline
  were wrong, five groups could not each land in their correct clinical band.
