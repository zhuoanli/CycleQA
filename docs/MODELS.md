# Models

Nothing is trained here. Every prediction already existed on disk; this work reads it.

## The system under audit (fixed)

**Dual-anchored MedSAM2.** MedSAM2 (hiera-tiny) is
prompted with a bounding box at ED and propagated in both temporal directions, prompted again at ES and likewise propagated
backward, and the two passes are merged at the temporal midpoint.

Each result NPZ therefore holds **three full-length predictions**:

| key | what it is |
|---|---|
| `ed_pred` | ED-anchored propagation (both directions), all T frames |
| `es_pred` | ES-anchored propagation (both directions), all T frames |
| `bidir` | the merged output — what the system reports today |

**`ed_pred` and `es_pred` are the entire basis of CycleQA.** They are two statistically
independent traversals of the same video, and their relationship is what CycleQA measures.

> **Prompts are derived from the reference masks**, so all results are an **oracle-prompt upper
> bound**. A deployed system additionally needs automatic prompt generation.

## Prediction directories

| Model | ACDC | M&Ms | Dual-anchored? | In CycleQA? |
|---|---|---|---|---|
| MedSAM2-dual (primary) | `results/medsam2/` | `results/medsam2_mnm/` | yes | **yes** |
| MedSAM2-ED / -ES | same NPZs, different key | same | yes | yes (reselection) |
| SAM2-dual | `results/sam2_bidir/` | `results/sam2_bidir_mnm/` | yes | yes (generalisation check) |
| U-Net (supervised) | — | `results/unet_mnm/` | no | not included — see below |
| DINOv2 (supervised) | — | `results/dinov2_mnm/` | no | not included |

## Why U-Net and DINOv2 are not included

1. They are frame-wise: with no temporal propagation there is no second anchor direction, and
   therefore no disagreement signal. Their NPZs contain only `bidir`, a single output. This is a
   property of frame-wise models rather than of this implementation, and it is why prior
   reference-free CMR quality-control work targeting such models (Robinson 2019, Tarroni 2019,
   Wang 2020, Hann 2021) does not use this signal.

2. Their training-checkpoint provenance could not be established. `infer_unet_mnm.py` was deleted
   in commit `c437e26`, and `results/unet_mnm/` predates both surviving checkpoints
   (`results/unet/best_model.pth`, ACDC-only; `results/unet_combined/best_model.pth`, trained on
   ACDC+M&Ms-2 and therefore not a clean external comparison). Metrics for both are retained in
   `results/patient_metrics.csv` but are not reported in the manuscript.
