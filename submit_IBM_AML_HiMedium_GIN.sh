#!/bin/bash
#PBS -N IBM_AML_HiMedium_GIN
#PBS -l select=1:ncpus=4:mem=64GB:ngpus=1:Qlist=ee:host=comp055
#PBS -q ee
#PBS -l walltime=200:00:00
#PBS -j oe
#PBS -m ae
#PBS -M 23724617@sun.ac.za
#PBS -V

DATASET="IBM_AML_HiMedium"
MODEL="GIN"
LOGFILE="${PBS_O_WORKDIR}/output_IBM_AML_HiMedium_GIN.out"
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

DATASET_DIR="Datasets/IBM_AML_dataset/HiMedium"

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
  echo "Starting ${DATASET} ${MODEL} on comp055 (actual: $(hostname))"
  export CUDA_VISIBLE_DEVICES=2
  python -u main.py "${DATASET}" "${MODEL}"
else
  echo "ERROR: missing training script"; ls -lah; exit 2
fi

echo "DONE"
