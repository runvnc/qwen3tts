#!/bin/bash
#
set FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST="12.0" pip install flash-attn --no-build-isolation
