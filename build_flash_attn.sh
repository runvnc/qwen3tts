#!/bin/bash
# Build flash-attention 2 for RTX 5090 (Blackwell, SM 12.0) with CUDA 12.8
#
# The default flash-attn build compiles for ALL GPU architectures, which takes
# forever (30-60+ min). Specifying only the target arch cuts it to ~5-10 min.
#
# RTX 5090 = GB202 = Blackwell = compute capability 12.0
#
# Usage:
#   chmod +x build_flash_attn.sh
#   ./build_flash_attn.sh
#
# Or to install a specific version:
#   FLASH_ATTN_VERSION=2.7.4 ./build_flash_attn.sh

set -euo pipefail

# ---- Configuration ----
# RTX 5090 (Blackwell) compute capability
export TORCH_CUDA_ARCH_LIST="12.0"

# Parallel build jobs - adjust based on RAM (each job uses ~2-4GB)
# Too many jobs can OOM during compilation
export MAX_JOBS=${MAX_JOBS:-$(nproc --ignore=2)}

# flash-attn version (empty = latest)
FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION:-""}

# Force build from source (skip wheel check)
export FLASH_ATTENTION_FORCE_BUILD=TRUE

# Skip tests during build
export FLASH_ATTENTION_SKIP_CUDA_BUILD_TEST=TRUE

# ---- Sanity checks ----
echo "=== Flash Attention 2 Builder for RTX 5090 ==="
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo "MAX_JOBS: ${MAX_JOBS}"
echo ""

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit 12.8+ first."
    echo "  apt install cuda-toolkit-12-8  (or similar)"
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[0-9.]+')
echo "CUDA version: ${NVCC_VERSION}"

# Check Python/torch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" 2>/dev/null || {
    echo "ERROR: PyTorch not found. Install PyTorch with CUDA 12.8 support first."
    exit 1
}

# Verify GPU is visible
python3 -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    cc = f'{props.major}.{props.minor}'
    print(f'GPU: {props.name}, Compute Capability: {cc}')
    if cc != '12.0':
        print(f'WARNING: Expected SM 12.0 (RTX 5090), got SM {cc}')
        print('         Adjust TORCH_CUDA_ARCH_LIST if building for a different GPU.')
else:
    print('WARNING: No CUDA GPU detected. Building anyway with specified arch.')
" 2>/dev/null || true

echo ""

# ---- Install build dependencies ----
echo "=== Installing build dependencies ==="
pip install --upgrade pip setuptools wheel packaging ninja

# ninja is critical - without it, build uses make which is much slower
if command -v ninja &> /dev/null; then
    echo "ninja found: $(ninja --version)"
else
    echo "WARNING: ninja not found, build will be slower"
fi

echo ""

# ---- Build flash-attn ----
echo "=== Building flash-attn (this takes ~5-10 min with correct arch) ==="
echo "    Started at: $(date)"
echo ""

BUILD_START=$(date +%s)

if [ -n "${FLASH_ATTN_VERSION}" ]; then
    echo "Installing flash-attn==${FLASH_ATTN_VERSION} from source..."
    pip install flash-attn==${FLASH_ATTN_VERSION} --no-build-isolation
else
    echo "Installing latest flash-attn from source..."
    pip install flash-attn --no-build-isolation
fi

BUILD_END=$(date +%s)
BUILD_SECS=$((BUILD_END - BUILD_START))
BUILD_MINS=$((BUILD_SECS / 60))

echo ""
echo "=== Build complete in ${BUILD_MINS}m ${BUILD_SECS}s ==="
echo ""

# ---- Verify ----
echo "=== Verifying installation ==="
python3 -c "
import flash_attn
print(f'flash-attn version: {flash_attn.__version__}')
from flash_attn import flash_attn_func
print('flash_attn_func imported successfully')
print('All good!')
"

echo ""
echo "Done. flash-attn is ready for RTX 5090."
