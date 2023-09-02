#!/bin/bash
list_job_cds=()

for _job_cd in "${list_job_cds[@]}"; do
  if [[ "$_job_cd" == "clm"* ]]; then
    python ./utils/run_clm.py --job_cd $_job_cd
  elif [[ "$_job_cd" == "loc"* ]]; then
    python ./utils/run_loc.py --job_cd $_job_cd
  elif [[ "$_job_cd" == "pft"* ]]; then
    python ./utils/run_pft.py --job_cd $_job_cd
  elif [[ "$_job_cd" == "prn"* ]]; then
    python ./utils/run_prn.py --job_cd $_job_cd
  elif [[ "$_job_cd" == "eval"* ]]; then
    python ./utils/run_eval.py --job_cd $_job_cd
  else
    echo "Wrong"
  fi
  echo "$_job_cd is executed"
  sleep 10
done