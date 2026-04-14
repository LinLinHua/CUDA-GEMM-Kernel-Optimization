# =========================================================================
# CUDA GEMM Project Build Manager
# -------------------------------------------------------------------------
# Targets:
#   make all        – build all src kernels + all tuning variants
#   make src        – build only src/ kernels
#   make tuning     – build only tuning/ variants
#   make profile    – run NCU on the ultimate kernel (roofline set)
#   make clean      – remove all compiled binaries
#
# Environment:
#   Tested on NVIDIA A100 (sm_80), CUDA 12.x
#   Project root: /root/workspace/00_GEMM_project/
# =========================================================================

NVCC        := nvcc
NVCC_FLAGS  := -O3 -arch=sm_80 -lcublas

SRC_DIR     := src
SRC_TARGETS := \
    01_gemm_naive       \
    02_gemm_tiled       \
    03_gemm_regblock    \
    04_gemm_float4      \
    05_gemm_cublas      \
    06_gemm_float4_ultimate

TUNING_DIR     := tuning
TUNING_TARGETS := \
    BK16_pad1       \
    BK16_pad2       \
    BK32_pad4       \
    BK16_pad2_lb2   \
    BM128_BK16_pad2

all: src tuning

src: $(SRC_TARGETS)
tuning: $(foreach t, $(TUNING_TARGETS), $(TUNING_DIR)/$(t)/kernel)

$(SRC_TARGETS): %: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(TUNING_DIR)/%/kernel: $(TUNING_DIR)/%/kernel.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

profile: 06_gemm_float4_ultimate
	ncu --force-overwrite \
	    --set roofline \
	    --kernel-name gemm_final \
	    -o results/06_gemm_float4_ultimate_result \
	    ./06_gemm_float4_ultimate 4096 4096 4096

profile_full: 06_gemm_float4_ultimate
	ncu --force-overwrite \
	    --set full \
	    --kernel-name gemm_final \
	    -o results/06_gemm_float4_ultimate_result_full \
	    ./06_gemm_float4_ultimate 4096 4096 4096

profile_bm128: tuning/BM128_BK16_pad2/kernel
	ncu --force-overwrite \
	    --set roofline \
	    --kernel-name gemm_final \
	    -o results/tuning_BM128_BK16_pad2 \
	    ./tuning/BM128_BK16_pad2/kernel 4096 4096 4096

clean:
	rm -f $(SRC_TARGETS)
	rm -f $(foreach t, $(TUNING_TARGETS), $(TUNING_DIR)/$(t)/kernel)