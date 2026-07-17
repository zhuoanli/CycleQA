# Environment

## The critical path needs no GPU

Every experiment in this paper is CPU-only NumPy/SciPy/scikit-learn. There is no training and
no inference — all predictions already exist as NPZ. Total runtime ≈ 20 min on 48 cores.

```bash
conda activate cinema_ft        # numpy, scipy, pandas, sklearn, matplotlib
python src/verify_data.py
python src/compute_metrics.py
python src/extract_features.py
python src/variant_features.py
python src/train_qc.py
python src/reselect.py
python src/make_figures.py
python src/make_tables.py
python src/verify_numbers.py
```


**`infer_medsam2.py` silently skips existing outputs.** If any input changes, pass
`--overwrite`. This bug once "processed" 2866 files in under a second, reusing stale
predictions.

**MedSAM2/SAM2 need a relative Hydra config** (`configs/sam2.1_hiera_t512.yaml`) and the job
must `--chdir` into `MedSAM2/`. Not needed for this paper — nothing here runs inference.


