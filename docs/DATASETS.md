# Datasets

Both cohorts are public. Neither is redistributed here; the scripts read the preprocessed
NPZ caches produced by the companion repository's preprocessing pipeline.

## ACDC — development (internal)

| | |
|---|---|
| Source | Automated Cardiac Diagnosis Challenge, Bernard et al. 2018, `10.1109/TMI.2018.2837502` |
| Patients | **100** (all used; the companion paper used a 20-patient validation subset) |
| Centre | 1 (University Hospital of Dijon) |
| **Vendor** | **Siemens only** — this single fact is what the paper turns on |
| Pathology | 5 groups × 20: NOR, DCM, HCM, MINF, RV |
| Reference masks | **ED and ES only.** No intermediate-phase annotation exists. |
| Role here | Develop CycleQA, then freeze. |

## M&Ms — external validation

| | |
|---|---|
| Source | Multi-Centre, Multi-Vendor & Multi-Disease Challenge, Campello et al. 2021, `10.1109/TMI.2021.3090082` |
| Patients | **320** with complete predictions (345 in the metadata CSV) |
| Vendors | Siemens 95 · Philips 125 · GE 50 · Canon 50 |
| Centres | 5 |
| Pathologies | 9: DCM, NOR, HCM, HHD, ARV, IHD, AHS, LVNC, Other |
| Reference masks | ED and ES only |
| Role here | **External validation only.** The risk model never sees an M&Ms study during development. |

Metadata: `MnM/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv`
(columns: `External code, VendorName, Vendor, Centre, ED, ES, Age, Pathology, Sex, Height, Weight`).

## Label convention — verified, not assumed

Both preprocessed caches use **1 = RV, 2 = Myocardium, 3 = LV**.

The raw M&Ms distribution uses the *opposite* convention (1 = LV, 3 = RV); this is remapped
during preprocessing. Rather than trust that, `src/verify_data.py` re-establishes it
**geometrically**: the LV is the blood pool whose boundary neighbourhood is ~100% myocardium
(measured: 1.000 on ACDC, 0.994 on M&Ms for label 3). Getting this wrong once produced an LV
Dice of 0.005.

## Gate 0a — the verification that had to pass first

Canon's degradation could have been an ED/ES indexing artefact rather than domain shift. If it
were, the whole vendor narrative would be false. Verified before any analysis:

| Check | Result |
|---|---|
| `ed_idx`/`es_idx` in NPZ == metadata `ED`/`ES`, **per vendor** | **100%** for Siemens, Philips, GE, Canon |
| GT LV volume at ED > at ES, per patient | **100%** for all four vendors |
| Label convention identical across cohorts | Confirmed geometrically |
| Disagreement↔error signal survives mid-cycle-only restriction | r = 0.745 (vs 0.756 all-frames) |

Run: `python src/verify_data.py` → exits non-zero on failure.
