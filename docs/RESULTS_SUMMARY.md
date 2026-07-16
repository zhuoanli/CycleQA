# Headline results

All from `results/*.csv`. Verify with `python src/verify_numbers.py`.

## 1. Benchmark accuracy does not transfer

| Cohort | n | RV | Myo | LV | **EF error** | **Failure rate** |
|---|---|---|---|---|---|---|
| ACDC (Siemens) | 100 | 0.913 | 0.801 | 0.858 | **6.4 pp** | **11%** |
| M&Ms Siemens | 95 | 0.885 | 0.749 | 0.816 | 5.7 pp | 28% |
| M&Ms Philips† | 125 | 0.743 | 0.674 | 0.697 | 29.0 pp | 66% |
| M&Ms GE† | 50 | 0.768 | 0.729 | 0.742 | 32.5 pp | 58% |
| M&Ms Canon† | 50 | 0.751 | 0.653 | 0.750 | **33.9 pp** | **70%** |
| **M&Ms all** | **320** | 0.790 | 0.702 | 0.748 | **23.4 pp** | **54%** |

† vendor absent from development.

**Dice degrades modestly. The clinical biomarker degrades sixfold.** Benchmark Dice understates
the damage.

## 2. The failure is visible with no ground truth

**Tier 1 — training-free.** Disagreement used *directly* as a Dice estimate:

| | r | MAE |
|---|---|---|
| ACDC | 0.735 | 0.064 |
| **M&Ms (external)** | **0.814** | **0.072** |

The signal is *stronger* out of domain than in it.

**Tier 2 — failure detection** (frozen on ACDC → applied once to M&Ms):

| Method | ACDC (out-of-fold) | M&Ms (external) |
|---|---|---|
| **Disagreement, no training** | 0.909 | **0.850** [0.811, 0.888] |
| Logistic regression | 0.915 | 0.851 [0.810, 0.892] |
| Random forest | 0.900 | 0.853 |

Per vendor: Siemens 0.883 · Philips 0.857 · GE 0.823 · **Canon 0.806**.

**Control — error detector, not vendor classifier:**

| | "is this Canon?" | "is this segmentation wrong?" |
|---|---|---|
| Image-OOD (appearance) | 0.478 | 0.573 |
| **CycleQA** | 0.636 | **0.851** |

## 3. Acting on it — no labels, no retraining, no extra inference

| Strategy | Dice | **EF error** | **Failure rate** |
|---|---|---|---|
| Current system (merged output) | 0.747 | **23.4 pp** | **54.4%** |
| Best fixed choice (hindsight only) | 0.799 | 16.8 pp | 21.9% |
| **CycleQA reselection** | **0.815** | **12.0 pp** | **14.7%** |
| Oracle (min EF error) | 0.812 | 10.4 pp | 17.8% |

CycleQA beats the best fixed strategy that could only be known with labels — because *no* fixed
strategy works everywhere. On Siemens it keeps the merged/ED output (5.8 pp; always-ES would
give 14.8 pp); on Canon/GE/Philips it switches to ES. **It adapts per patient and per domain
without ever being told the vendor.**

## Ablation

| Feature set | Internal (out-of-fold) | External |
|---|---|---|
| Anchor agreement, fitted (24 features) | 0.987 | 0.793 |
| Plausibility only (22) | 0.863 | 0.891 |
| All families (46) | 0.915 | 0.851 |
| **Raw disagreement, no fitting** | 0.909 | **0.850** |

The raw disagreement signal transfers without any fitting, while learned combinations are more
sensitive to feature selection and to the limited number of development failures (11 of 100).
The training-free tier is therefore the recommended configuration.
