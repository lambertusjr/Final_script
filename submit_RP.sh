#!/bin/bash
#PBS -N Fraud_GNN_IBM_Medium
#PBS -l select=1:ncpus=4:mem=64GB:ngpus=1:Qlist=ee:host=comp056
#PBS -q ee
#PBS -l walltime=200:00:00
#PBS -j oe
#PBS -o output.out
#PBS -m ae
#PBS -M 23724617@sun.ac.za
#PBS -V

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
  /usr/bin/rsync -vax --progress \
    --include '/csv_results/***' \
    --include '/optimization_results.db' \
    --include '/output.out' \
    --include '/worker*.log' \
    --exclude '*' \
    "${TMP}/" "${PBS_O_WORKDIR}/" || true
  [ "$?" -eq 0 ] && /bin/rm -rf "${TMP}"
}
trap cleanup EXIT

echo "Copying from ${PBS_O_WORKDIR}/ to ${TMP}/"
/usr/bin/rsync -vax --delete "${PBS_O_WORKDIR}/" "${TMP}/"
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

# threads consistent with ncpus=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export QT_QPA_PLATFORM=offscreen
export MPLCONFIGDIR="${TMP}/.mpl"
mkdir -p "${MPLCONFIGDIR}"

python -c "import torch, sys; print('torch', torch.__version__, 'cuda', getattr(torch.version,'cuda',None), 'cuda_available', torch.cuda.is_available())"

if [[ -f main.py ]]; then
  echo "Starting Worker 0 on GPU 0 (LiMedium)"
  # Run in background with & and redirect output
  python -u main.py IBM_AML_LiMedium GCN
  
  # Wait for all background jobs to finish before cleanup
  wait
else
  echo "ERROR: missing training script"; ls -lah; exit 2
fi

echo "DONE"

