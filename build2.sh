#!/bin/bash
#PBS -N build_RP_env
#PBS -l select=1:ncpus=4:mem=16GB
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o build_env.out
#PBS -V

set -euxo pipefail

# Define correct PyG Wheel URL (Use 2.4.0 for all 2.4.x versions)
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-2.4.0+cu118.html"

umask 0077
cd "$PBS_O_WORKDIR"
TMP="/scratch-small-local/${PBS_JOBID//./-}"
mkdir -p "$TMP"

# 1. Setup Micromamba
cp "$PBS_O_WORKDIR/bin/micromamba" "$TMP/micromamba"
chmod +x "$TMP/micromamba"
MAMBA_EXE="$TMP/micromamba"
export MAMBA_ROOT_PREFIX="$TMP/micromamba_root"

# 2. Create Base Environment (PyTorch + Core Libs)
echo ">>> Creating Environment..."
$MAMBA_EXE create -y --no-rc -p "$TMP/RP_env" -f environment.yaml

# 3. Install PyTorch Geometric Dependencies
# We use the correct URL to get binaries compatible with older Cluster OS
RP_PIP="$TMP/RP_env/bin/pip"

echo ">>> Installing PyG binaries from $PYG_WHEEL_URL"
$RP_PIP install torch_geometric

# Note: We explicitly DO NOT install pyg_lib if it causes GLIBC issues.
# torch-sparse is sufficient for NeighborSampler.
$RP_PIP install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f "$PYG_WHEEL_URL" \
    --no-index  # Forces pip to fail if it can't find wheels at the URL (prevents PyPI fallback)

# 4. Pack the Environment
echo ">>> Packing Environment..."
$MAMBA_EXE install -y -p "$TMP/RP_env" conda-pack
$TMP/RP_env/bin/conda-pack -p "$TMP/RP_env" -o "$PBS_O_WORKDIR/RP_env.tar.gz" --force

# Cleanup
rm -rf "$TMP"
echo ">>> SUCCESS: Environment rebuilt."