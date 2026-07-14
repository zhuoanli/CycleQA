#!/bin/bash
# Live CycleQA pipeline progress.   Usage:  bash metis/jobs/watch.sh   (Ctrl-C to exit)
cd /scratch/gautschi/li4533/MIUA_2026
watch -n 2 --color '
printf "\033[1m=== RUNNING ===\033[0m\n"
for j in compute_metrics extract_features train_qc variant_features make_figures; do
  n=$(ps -u $USER -o cmd= | grep -c "[m]etis/src/$j" )
  [ "$n" -gt 0 ] && printf "  \033[32m%-18s %3d procs\033[0m\n" "$j" "$n"
done
[ "$(ps -u $USER -o cmd= | grep -c "[m]etis/src/")" -eq 0 ] && printf "  (idle)\n"

printf "\n\033[1m=== LOGS ===\033[0m\n"
for f in metis/results/log_*.txt; do
  [ -f "$f" ] || continue
  printf "  \033[36m%s\033[0m\n" "$(basename $f)"
  tail -2 "$f" | sed "s/^/    /"
done

printf "\n\033[1m=== OUTPUTS ===\033[0m\n"
for f in patient_metrics patient_features variant_features dice_estimation qc_performance \
         ood_control risk_coverage ablation failure_prevalence qc_scores reselection; do
  p=metis/results/$f.csv
  if [ -f "$p" ]; then printf "  \033[32m[x]\033[0m %-22s %s\n" "$f" "$(wc -l < $p) rows"
  else printf "  [ ] %-22s\n" "$f"; fi
done
'
