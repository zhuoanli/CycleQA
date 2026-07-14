#!/bin/bash
# Regenerate every result, figure and table from the raw prediction NPZs.
# CPU only. No GPU, no training, no inference. ~20 min on 48 cores.
set -e
cd "$(dirname "$0")/../.."          # -> MIUA_2026
export CYCLEQA_PROCS="${CYCLEQA_PROCS:-48}"
S=metis/src

echo "== Gate 0a: data verification (BLOCKING) =="
python $S/verify_data.py

echo "== Gate 0b: per-patient reference metrics =="
python $S/compute_metrics.py

echo "== Phase 1: no-reference CycleQA features =="
python $S/extract_features.py
python $S/variant_features.py

echo "== Phase 2: quality assurance =="
python $S/train_qc.py
python $S/reselect.py

echo "== Phase 3: figures and tables =="
python $S/make_figures.py
python $S/make_tables.py

echo "== Verification: every manuscript number must trace to a CSV =="
python $S/verify_numbers.py

echo "== done =="
