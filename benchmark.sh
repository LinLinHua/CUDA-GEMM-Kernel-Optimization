#!/bin/bash

# =========================================================================
# GEMM Performance Benchmarking Suite
# -------------------------------------------------------------------------
# This script executes all GEMM versions across different matrix sizes
# and formats the results into a Markdown table for performance analysis.
# =========================================================================

# List of implementations to test (ordered by optimization level)
SRC_METHODS="01_gemm_naive 02_gemm_tiled 03_gemm_regblock 04_gemm_float4 06_gemm_float4_ultimate 05_gemm_cublas"
TUNING_METHODS="BK16_pad1 BK16_pad2 BK32_pad4 BK16_pad2_lb2 BM128_BK16_pad2"
# Matrix dimensions to benchmark (M=N=K)
SIZES="1024 2048 4096"

echo "Running benchmarks..."

# Print Markdown Table Header
echo -e "\n## src kernels\n"
echo -e "\n| Matrix Size | Implementation | Execution Time (ms) | Throughput (TFLOPS) |"
echo "| :--- | :--- | :--- | :--- |"

for size in $SIZES; do
    for method in $SRC_METHODS; do
        if [ -f "/workspace/$method" ]; then
            RESULT=$(./$method $size 2>/dev/null)
            TIME=$(echo "$RESULT"   | grep "Time"       | awk '{print $3}')
            TFLOPS=$(echo "$RESULT" | grep "Throughput" | awk '{print $2}')
            echo "| $size x $size | $method | $TIME | $TFLOPS |"
        fi
    done
    echo "| --- | --- | --- | --- |"
done

echo -e "\n## tuning kernels\n"
echo "| Matrix Size | Implementation | Execution Time (ms) | Throughput (TFLOPS) |"
echo "| :--- | :--- | :--- | :--- |"

for size in $SIZES; do
    for method in $TUNING_METHODS; do
        if [ -f "/workspace/tuning/$method/kernel" ]; then
            RESULT=$(./tuning/$method/kernel $size 2>/dev/null)
            TIME=$(echo "$RESULT"   | grep "Time"       | awk '{print $3}')
            TFLOPS=$(echo "$RESULT" | grep "Throughput" | awk '{print $2}')
            echo "| $size x $size | $method | $TIME | $TFLOPS |"
        fi
    done
    echo "| --- | --- | --- | --- |"
done

# Reference: IEEE Std 1003.1 (POSIX Shell)
# "Automation of performance critical tasks ensures reproducibility 
# of scientific computing benchmarks."