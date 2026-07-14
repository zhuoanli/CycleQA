# Models

Nothing is trained here. Every prediction already existed on disk; this work reads it.

## The system under audit (fixed)

**Dual-anchored MedSAM2** — the companion MIUA main-track paper. MedSAM2 (hiera-tiny) is
prompted with a bounding box at ED and propagated forward, prompted again at ES and propagated
backward, and the two passes are merged at the temporal midpoint.

Each result NPZ therefore holds **three full-length predictions**:

| key | what it is |
|---|---|
| `ed_pred` | forward propagation from the ED anchor, all T frames |
| `es_pred` | backward propagation from the ES anchor, all T frames |
| `bidir` | the merged output — what the system reports today |

**`ed_pred` and `es_pred` are the entire basis of CycleQA.** They are two statistically
independent traversals of the same video. Nobody had looked at the relationship *between* them.

> **Prompts are derived from the reference masks.** All results are therefore an
> **oracle-prompt upper bound**. A deployed system additionally needs automatic prompt
> generation. Say this in any talk about this work.

## Prediction directories

| Model | ACDC | M&Ms | Dual-anchored? | In CycleQA? |
|---|---|---|---|---|
| MedSAM2-dual (primary) | `results/medsam2/` | `results/medsam2_mnm/` | yes | **yes** |
| MedSAM2-ED / -ES | same NPZs, different key | same | yes | yes (reselection) |
| SAM2-dual | `results/sam2_bidir/` | `results/sam2_bidir_mnm/` | yes | yes (generalisation check) |
| U-Net (supervised) | — | `results/unet_mnm/` | **no** | **excluded — see below** |
| DINOv2 (supervised) | — | `results/dinov2_mnm/` | **no** | **excluded** |

## Why U-Net and DINOv2 are excluded

Two reasons, and the first is the one that matters scientifically:

1. **They are frame-wise. They have no propagation, therefore no second direction, therefore
   no disagreement signal.** Their NPZs contain only `bidir` — a single output. This is not a
   limitation of our implementation; it is the structural point of the paper. A frame-wise CNN
   *cannot* produce this signal, which is why prior CMR quality-control work
   (Robinson 2019, Tarroni 2019, Wang 2020) could not have found it.

2. Their checkpoint provenance is unresolved. `infer_unet_mnm.py` was deleted in commit
   `c437e26`, and `results/unet_mnm/` predates both surviving checkpoints
   (`results/unet/best_model.pth`, ACDC-only; `results/unet_combined/best_model.pth`,
   ACDC+M&Ms-2 — which would leak). We report metrics for them in
   `results/patient_metrics.csv` but keep them out of the manuscript rather than defend a
   comparison we cannot substantiate.
