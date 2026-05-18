#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32GB:ngpus=1:Qlist=ee:host=comp055
#PBS -q ee
#PBS -l walltime=200:00:00
#PBS -j oe
#PBS -m ae
#PBS -M 23724617@sun.ac.za
#PBS -V

# PBS Pro doesn't support -F (script args). Pass via -v at qsub time instead.
# Usage:
#   qsub -N "AMLSim_GCN" \
#        -l "select=1:ncpus=4:mem=32GB:ngpus=1:Qlist=ee:host=comp056" \
#        -v "DATASET=AMLSim,MODEL=GCN,NODE=56,GPU=0" \
#        submit_RP.sh
# (launch.sh builds this for you.)
: "${DATASET:?DATASET env var required (pass via qsub -v ...)}"
: "${MODEL:?MODEL env var required (pass via qsub -v ...)}"
: "${NODE:?NODE env var required (pass via qsub -v ...)}"
: "${GPU:?GPU env var required (pass via qsub -v ...)}"

# Pin to a specific GPU index on the chosen node.
export CUDA_VISIBLE_DEVICES="${GPU}"

# Unique suffix for result/cache filenames so concurrent jobs don't clobber
# each other on rsync push. Dots in PBS_JOBID confuse os.path.splitext.
export JOB_ID="${PBS_JOBID//./-}"

LOGFILE="${PBS_O_WORKDIR}/output_${DATASET}_${MODEL}_gpu${GPU}.out"
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
    --include='batch_size_cache_*.json' \
    --exclude='*' \
    "${TMP}/" "${PBS_O_WORKDIR}/"; then
    /bin/rm -rf "${TMP}"
  fi
}
trap cleanup EXIT

# Map DATASET arg to its subdirectory under Datasets/
case "${DATASET}" in
  Elliptic)       DATASET_DIR="Datasets/Elliptic_dataset" ;;
  IBM_AML_HiSmall)  DATASET_DIR="Datasets/IBM_AML_dataset/HiSmall" ;;
  IBM_AML_LiSmall)  DATASET_DIR="Datasets/IBM_AML_dataset/LiSmall" ;;
  IBM_AML_HiMedium) DATASET_DIR="Datasets/IBM_AML_dataset/HiMedium" ;;
  IBM_AML_LiMedium) DATASET_DIR="Datasets/IBM_AML_dataset/LiMedium" ;;
  AMLSim)         DATASET_DIR="Datasets/AMLSim_dataset" ;;
  *) echo "ERROR: unknown DATASET '${DATASET}'"; exit 1 ;;
esac

echo "Copying code from ${PBS_O_WORKDIR}/ to ${TMP}/ (excluding all Datasets)"
/usr/bin/rsync -vax --delete \
  --exclude 'RP_env' \
  --exclude '__pycache__' \
  --exclude 'output_*.out' \
  --exclude 'Datasets/' \
  "${PBS_O_WORKDIR}/" "${TMP}/"

echo "Copying only ${DATASET_DIR} to ${TMP}/"
mkdir -p "${TMP}/${DATASET_DIR}"
/usr/bin/rsync -vax \
  "${PBS_O_WORKDIR}/${DATASET_DIR}/" "${TMP}/${DATASET_DIR}/"

cd "${TMP}"

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

# prebuilt env (extract into its own subdir so activate path exists)
mkdir -p "${TMP}/RP_env"
tar -xzf "${TMP}/RP_env.tar.gz" -C "${TMP}/RP_env"
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
  echo "Starting ${DATASET} ${MODEL} on comp0${NODE}:gpu${GPU} (actual: $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, JOB_ID=${JOB_ID})"
  python -u main.py "${DATASET}" "${MODEL}"
else
  echo "ERROR: missing training script"; ls -lah; exit 2
fi

echo "DONE"

