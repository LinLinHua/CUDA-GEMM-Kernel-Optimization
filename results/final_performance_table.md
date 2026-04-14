Running benchmarks...

## src kernels


| Matrix Size | Implementation | Execution Time (ms) | Throughput (TFLOPS) |
| :--- | :--- | :--- | :--- |
| 1024 x 1024 | 01_gemm_naive | 0.885 | 2.43 |
| 1024 x 1024 | 02_gemm_tiled | 0.539 | 3.99 |
| 1024 x 1024 | 03_gemm_regblock | 0.481 | 4.47 |
| 1024 x 1024 | 04_gemm_float4 | 0.336 | 6.38 |
| 1024 x 1024 | 06_gemm_float4_ultimate | 0.317 | 6.77 |
| 1024 x 1024 | 05_gemm_cublas | 0.148 | 14.54 |
| --- | --- | --- | --- |
| 2048 x 2048 | 01_gemm_naive | 6.151 | 2.79 |
| 2048 x 2048 | 02_gemm_tiled | 3.608 | 4.76 |
| 2048 x 2048 | 03_gemm_regblock | 2.683 | 6.40 |
| 2048 x 2048 | 04_gemm_float4 | 2.579 | 6.66 |
| 2048 x 2048 | 06_gemm_float4_ultimate | 1.948 | 8.82 |
| 2048 x 2048 | 05_gemm_cublas | 1.247 | 13.77 |
| --- | --- | --- | --- |
| 4096 x 4096 | 01_gemm_naive | 57.221 | 2.40 |
| 4096 x 4096 | 02_gemm_tiled | 25.777 | 5.33 |
| 4096 x 4096 | 03_gemm_regblock | 13.428 | 10.23 |
| 4096 x 4096 | 04_gemm_float4 | 13.788 | 9.97 |
| 4096 x 4096 | 06_gemm_float4_ultimate | 12.094 | 11.36 |
| 4096 x 4096 | 05_gemm_cublas | 7.218 | 19.04 |
| --- | --- | --- | --- |

## tuning kernels

| Matrix Size | Implementation | Execution Time (ms) | Throughput (TFLOPS) |
| :--- | :--- | :--- | :--- |
| 1024 x 1024 | BK16_pad1 | 0.313 | 6.86 |
| 1024 x 1024 | BK16_pad2 | 0.317 | 6.77 |
| 1024 x 1024 | BK32_pad4 | 0.321 | 6.68 |
| 1024 x 1024 | BK16_pad2_lb2 | 0.291 | 7.37 |
| 1024 x 1024 | BM128_BK16_pad2 | 0.291 | 7.37 |
| --- | --- | --- | --- |
| 2048 x 2048 | BK16_pad1 | 1.884 | 9.12 |
| 2048 x 2048 | BK16_pad2 | 2.029 | 8.47 |
| 2048 x 2048 | BK32_pad4 | 2.038 | 8.43 |
| 2048 x 2048 | BK16_pad2_lb2 | 2.005 | 8.57 |
| 2048 x 2048 | BM128_BK16_pad2 | 2.008 | 8.56 |
| --- | --- | --- | --- |
| 4096 x 4096 | BK16_pad1 | 14.381 | 9.56 |
| 4096 x 4096 | BK16_pad2 | 10.165 | 13.52 |
| 4096 x 4096 | BK32_pad4 | 12.490 | 11.00 |
| 4096 x 4096 | BK16_pad2_lb2 | 9.996 | 13.75 |
| 4096 x 4096 | BM128_BK16_pad2 | 9.997 | 13.75 |
| --- | --- | --- | --- |
