#!/bin/bash
# Build flash-attention 2 for RTX 5090 (Blackwell, SM 120) with CUDA 12.8
#
# IMPORTANT: flash-attn uses FLASH_ATTN_CUDA_ARCHS (not TORCH_CUDA_ARCH_LIST!)
# Without this, it builds for ALL architectures = 30-60+ min.
# With it, ~5-10 min.
#
# See: https://github.com/Dao-AILab/flash-attention/issues/1560
#
# NOTE: As of early 2026, flash-attn Blackwell (sm_120) support is
# still maturing. FA2 kernels may compile but crash at runtime.
# If that happens, fall back to sdpa (PyTorch built-in).
# The server_optimized.py already handles this automatically.
#
# Usage:
#   chmod +x build_flash_attn.sh
#   ./build_flash_attn.sh

set -euo pipefail

# ---- Configuration ----
# RTX 5090 = Blackwell = SM 120
# This is the flash-attn specific env var (NOT TORCH_CUDA_ARCH_LIST!)
export FLASH_ATTN_CUDA_ARCHS="120"

# Also set TORCH_CUDA_ARCH_LIST for any PyTorch extension builds
export TORCH_CUDA_ARCH_LIST="12.0"

# Parallel build jobs - adjust based on RAM (each job uses ~2-4GB)
export MAX_JOBS=${MAX_JOBS:-$(nproc --ignore=2)}

# Force build from source
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD_TEST=TRUE

# ---- Sanity checks ----
echo "=== Flash Attention 2 Builder for RTX 5090 ==="
echo "FLASH_ATTN_CUDA_ARCHS: ${FLASH_ATTN_CUDA_ARCHS}"
echo "TORCH_CUDA_ARCH_LIST:  ${TORCH_CUDA_ARCH_LIST}"
echo "MAX_JOBS:              ${MAX_JOBS}"
echo ""

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit 12.8+ first."
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[0-9.]+')
echo "CUDA version: ${NVCC_VERSION}"

# Check Python/torch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" 2>/dev/null || {
    echo "ERROR: PyTorch not found."
    exit 1
}

# Check GPU
python3 -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}, SM {props.major}.{props.minor}')
else:
    print('WARNING: No CUDA GPU detected.')
" 2>/dev/null || true

echo ""

# ---- Install build deps ----
echo "=== Installing build dependencies ==="
pip install --upgrade pip setuptools wheel packaging ninja
echo ""

# ---- Build ----
echo "=== Building flash-attn for SM 120 only ==="
echo "    Started at: $(date)"
echo ""

BUILD_START=$(date +%s)

pip install flash-attn --no-build-isolation

BUILD_END=$(date +%s)
BUILD_SECS=$((BUILD_END - BUILD_START))
BUILD_MINS=$((BUILD_SECS / 60))
BUILD_REM=$((BUILD_SECS % 60))

echo ""
echo "=== Build complete in ${BUILD_MINS}m ${BUILD_REM}s ==="
echo ""

# ---- Verify ----
echo "=== Verifying ==="
python3 -c "
import flash_attn
print(f'flash-attn version: {flash_attn.__version__}')
try:
    from flash_attn import flash_attn_func
    print('flash_attn_func imported OK')
except Exception as e:
    print(f'Import warning: {e}')
print()
print('NOTE: If you get runtime CUDA errors on sm_120,')  
print('Blackwell kernel support may still be incomplete.')  
print('The server will auto-fallback to sdpa in that case.')
"
