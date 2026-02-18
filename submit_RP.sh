#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32GB:ngpus=1:Qlist=ee:host=comp056
#PBS -q ee
#PBS -l walltime=200:00:00
#PBS -j oe
#PBS -m ae
#PBS -M 23724617@sun.ac.za
#PBS -V

# Usage: qsub -F "AMLSim GAT" submit_RP.sh
DATASET="${1:?Usage: qsub -F \"DATASET MODEL\" submit_RP.sh}"
MODEL="${2:?Usage: qsub -F \"DATASET MODEL\" submit_RP.sh}"

# PBS directives are parsed before $1/$2 are available, so set the job name
# at runtime and redirect all output to a job-specific log file manually.
qalter -N "${DATASET}_${MODEL}" "${PBS_JOBID}" 2>/dev/null || true
LOGFILE="${PBS_O_WORKDIR}/output_${DATASET}_${MODEL}.out"
exec > "${LOGFILE}" 2>&1

set -euxo pipefail

umask 0077
SCRATCH_BASE="/scratch-small-local"
[ -d "${SCRATCH_BASE}" ] || SCRATCH_BASE="$HOME/scratch"
TMP="${SCRATCH_BASE}/${PBS_JOBID//./-}"
mkdir -p "${TMP}"
echo "Temporary work dir: ${TMP}"


cd ${TMP}

cleanup() {
  echo "Copying results back to ${PBS_O_WORKDIR}/ (cleanup)"
  if /usr/bin/rsync -vax --progress \
    --include='results/' \
    --include='results/**' \
    --include='optimization_results_on_*.db' \
    --include='batch_size_cache.json' \
    --exclude='*' \
    "${TMP}/" "${PBS_O_WORKDIR}/"; then
    /bin/rm -rf "${TMP}"
  fi
}
trap cleanup EXIT

echo "Copying from ${PBS_O_WORKDIR}/ to ${TMP}/"
/usr/bin/rsync -vax --delete \
  --exclude 'RP_env.tar.gz' \
  --exclude 'RP_env' \
  --exclude '__pycache__' \
  --exclude 'output_*.out' \
  "${PBS_O_WORKDIR}/" "${TMP}/"
cd "${TMP}"

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

# prebuilt env (extract into its own subdir so activate path exists)
mkdir -p "${TMP}/RP_env"
tar -xzf "$PBS_O_WORKDIR/RP_env.tar.gz" -C "${TMP}/RP_env"
# conda-pack activate references some unset vars under set -u; relax then restore
set +u
source "${TMP}/RP_env/bin/activate"
command -v conda-unpack >/dev/null 2>&1 && conda-unpack || true
set -u

# threads consistent with ncpus=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export QT_QPA_PLATFORM=offscreen
export MPLCONFIGDIR="${TMP}/.mpl"
mkdir -p "${MPLCONFIGDIR}"

python -c "import torch, sys; print('torch', torch.__version__, 'cuda', getattr(torch.version,'cuda',None), 'cuda_available', torch.cuda.is_available())"

if [[ -f main.py ]]; then
  echo "Starting ${DATASET} ${MODEL} on $(hostname)"
  python -u main.py "${DATASET}" "${MODEL}"
else
  echo "ERROR: missing training script"; ls -lah; exit 2
fi

echo "DONE"

