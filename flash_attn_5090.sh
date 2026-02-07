#!/bin/bash
#
FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="12.0" pip install flash-attn --verbose --no-build-isolation
