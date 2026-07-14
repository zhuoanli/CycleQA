# Environment

## The critical path needs no GPU

Every experiment in this paper is CPU-only NumPy/SciPy/scikit-learn. There is no training and
no inference — all predictions already exist as NPZ. Total runtime ≈ 20 min on 48 cores.

```bash
module load conda
conda activate cinema_ft        # numpy, scipy, pandas, sklearn, matplotlib
cd /scratch/gautschi/li4533/MIUA_2026
bash metis/jobs/run_all.sh
```

`CYCLEQA_PROCS` overrides the worker count (default 48).

## Cluster

Purdue Gautschi, SLURM, account `life`. Partitions: `cpu` (what this needs), `ai` (H100),
`smallgpu` (L40S). The login node has 384 cores and no GPU — `nvidia-smi` failing there is
expected, not a bug.

## Traps that cost real time

**Do not build the full `(T, Z, 512, 512)` stack.** The first version of
`extract_features.py` did, at ~250 MB per worker; with 48 workers it thrashed and the ETA hit
8 hours. Masks are streamed slice by slice and reduced on the fly — a 3D Dice is a ratio of
summed voxel counts, so it accumulates exactly across slices. Each slice is also cropped to
its bounding box (the heart fills a small part of the frame). Peak memory dropped to ~25 MB
per worker and the runtime to 1 minute.

**`infer_medsam2.py` silently skips existing outputs.** If any input changes, pass
`--overwrite`. This bug once "processed" 2866 files in under a second, reusing stale
predictions.

**MedSAM2/SAM2 need a relative Hydra config** (`configs/sam2.1_hiera_t512.yaml`) and the job
must `--chdir` into `MedSAM2/`. Not needed for this paper — nothing here runs inference.

**`cinema_ft` is cu130** and fails on `smallgpu` (CUDA 12.9 driver). `train_gpu_env` (cu124) is
the GPU-side env. Irrelevant here; noted so nobody rediscovers it.

## No LaTeX on the login node

Compile `paper/main.tex` on Overleaf.
