#!/usr/bin/env bash
# Submit one or many PBS jobs via the templated submit_RP.sh script.
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
# NODE is the numeric suffix only (e.g. 55 -> comp055); submit_RP.sh prepends "comp0".

set -euo pipefail
SCRIPT="submit_RP.sh"

# Normalize args to "DATASET MODEL NODE GPU" regardless of NODE GPU vs NODE:GPU input.
normalize() {
  local dataset="$1" model="$2" third="$3" fourth="${4:-}"
  if [[ "$third" == *:* ]]; then
    local node="${third%%:*}" gpu="${third##*:}"
    echo "$dataset $model $node $gpu"
  else
    [[ -z "$fourth" ]] && { echo "ERROR: missing GPU arg for: $dataset $model $third" >&2; return 1; }
    echo "$dataset $model $third $fourth"
  fi
}

submit_one() {
  local args; args=$(normalize "$@")
  echo "qsub -F \"$args\" $SCRIPT"
  # shellcheck disable=SC2086
  qsub -F "$args" "$SCRIPT"
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
