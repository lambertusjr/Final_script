#!/usr/bin/env bash
# Submit one or many PBS Pro jobs via the templated submit_RP.sh script.
#
# Usage:
#   ./launch.sh DATASET MODEL NODE GPU         # single job, e.g. AMLSim GCN 55 0
#   ./launch.sh DATASET MODEL NODE:GPU         # compact form, e.g. AMLSim GCN 55:0
#   ./launch.sh -f jobs.txt                    # batch from manifest
#
# Manifest format (one job per line):
#   DATASET MODEL NODE GPU       or       DATASET MODEL NODE:GPU
# Blank lines and lines starting with '#' are ignored.
#
# NODE is the numeric suffix only (e.g. 55 -> comp055).
# PBS Pro doesn't support -F; args are passed via -v (env vars) and -l (host).

set -euo pipefail
SCRIPT="submit_RP.sh"

# Resource string: matches submit_RP.sh's #PBS -l select. Edit here if your
# default resource needs change (or pass --select="..." in the future).
SELECT_TEMPLATE='select=1:ncpus=4:mem=32GB:ngpus=1:Qlist=ee:host=comp0%s'

# Parse "DATASET MODEL NODE GPU" or "DATASET MODEL NODE:GPU".
parse_args() {
  local dataset="$1" model="$2" third="$3" fourth="${4:-}"
  if [[ "$third" == *:* ]]; then
    NODE="${third%%:*}"
    GPU="${third##*:}"
  else
    [[ -z "$fourth" ]] && { echo "ERROR: missing GPU arg for: $dataset $model $third" >&2; return 1; }
    NODE="$third"
    GPU="$fourth"
  fi
  DATASET="$dataset"
  MODEL="$model"
}

submit_one() {
  parse_args "$@"
  local select_str
  # shellcheck disable=SC2059
  select_str=$(printf "$SELECT_TEMPLATE" "$NODE")
  local jobname="${DATASET}_${MODEL}"
  local varlist="DATASET=${DATASET},MODEL=${MODEL},NODE=${NODE},GPU=${GPU}"

  echo "qsub -N $jobname -l $select_str -v $varlist $SCRIPT"
  qsub \
    -N "$jobname" \
    -l "$select_str" \
    -v "$varlist" \
    "$SCRIPT"
}

if [[ "${1:-}" == "-f" ]]; then
  manifest="${2:?Usage: $0 -f manifest.txt}"
  [[ -r "$manifest" ]] || { echo "ERROR: cannot read $manifest" >&2; exit 1; }
  while IFS= read -r line || [[ -n "$line" ]]; do
    # strip comments and trim
    line="${line%%#*}"
    line="$(echo "$line" | xargs || true)"
    [[ -z "$line" ]] && continue
    # shellcheck disable=SC2086
    submit_one $line
  done < "$manifest"
else
  [[ $# -lt 3 ]] && {
    echo "Usage: $0 DATASET MODEL NODE GPU"
    echo "       $0 DATASET MODEL NODE:GPU"
    echo "       $0 -f manifest.txt"
    exit 1
  }
  submit_one "$@"
fi
