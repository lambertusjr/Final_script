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

$RP_PIP install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f "$PYG_WHEEL_URL" \
    --no-index

# Optional: pyg-lib provides the fastest C++ NeighborLoader sampler (2-5x over
# torch-sparse). Install attempt is non-fatal — if the cluster GLIBC is too old
# for the prebuilt wheel, the job continues and code falls back to torch-sparse
# automatically.
echo ">>> Attempting pyg_lib install (optional — sampling still works without it)"
$RP_PIP install pyg_lib \
    -f "$PYG_WHEEL_URL" \
    --no-index \
    || echo "WARNING: pyg_lib install failed (likely GLIBC < 2.28). Continuing with torch-sparse sampler."

# Log which sampler backend will be active.
# Catch OSError too: on old GLIBC clusters the wheel installs but libpyg.so
# fails to dlopen, which raises OSError rather than ImportError.
# If pyg_lib is broken, uninstall it so the packed env doesn't carry a dud package.
$TMP/RP_env/bin/python -c "
import sys, subprocess
try:
    import pyg_lib
    print('SAMPLER BACKEND: pyg-lib', pyg_lib.__version__, '(C++ fast path)')
except (ImportError, OSError) as e:
    print('WARNING: pyg_lib not usable (' + str(e) + '). Uninstalling.')
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'pyg_lib'], check=False)
    import torch_sparse
    print('SAMPLER BACKEND: torch-sparse', torch_sparse.__version__, '(fallback)')
"

# 4. Pack the Environment
echo ">>> Packing Environment..."
$MAMBA_EXE install -y -p "$TMP/RP_env" conda-pack
$TMP/RP_env/bin/conda-pack -p "$TMP/RP_env" -o "$PBS_O_WORKDIR/RP_env.tar.gz" --force

# Cleanup
rm -rf "$TMP"
echo ">>> SUCCESS: Environment rebuilt."