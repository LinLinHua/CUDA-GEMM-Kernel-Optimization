#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8

__global__ void __launch_bounds__(64, 4)
gemm_final(const float* A, const float* B, float* C, int M, int N, int K) {

    // padding +4 eliminate bank conflict
    __shared__ float sA[BM][BK + 4];
    __shared__ float sB[BK][BN + 4];

    int threadRow = threadIdx.x / (BN / TN);
    int threadCol = threadIdx.x % (BN / TN);
    int blockRowStart = blockIdx.y * BM;
    int blockColStart = blockIdx.x * BN;
    int totalThreads = (BM / TM) * (BN / TN);  // 64
    int tid = threadIdx.x;

    float regC[TM][TN] = {0.0f};

    for (int k = 0; k < K; k += BK) {

        // float4 load A tile
        for (int i = tid; i < (BM * BK) / 4; i += totalThreads) {
            int row = i / (BK / 4);
            int col = (i % (BK / 4)) * 4;
            int globalRow = blockRowStart + row;
            int globalCol = k + col;
            if (globalRow < M && globalCol + 3 < K) {
                float4 tmp = reinterpret_cast<const float4*>(&A[globalRow * K + globalCol])[0];
                sA[row][col]     = tmp.x;
                sA[row][col + 1] = tmp.y;
                sA[row][col + 2] = tmp.z;
                sA[row][col + 3] = tmp.w;
            } else {
                for (int j = 0; j < 4; j++)
                    sA[row][col + j] = (globalRow < M && globalCol + j < K)
                                       ? A[globalRow * K + globalCol + j] : 0.0f;
            }
        }

        // float4 load B tile
        for (int i = tid; i < (BK * BN) / 4; i += totalThreads) {
            int row = i / (BN / 4);
            int col = (i % (BN / 4)) * 4;
            int globalRow = k + row;
            int globalCol = blockColStart + col;
            if (globalRow < K && globalCol + 3 < N) {
                float4 tmp = reinterpret_cast<const float4*>(&B[globalRow * N + globalCol])[0];
                sB[row][col]     = tmp.x;
                sB[row][col + 1] = tmp.y;
                sB[row][col + 2] = tmp.z;
                sB[row][col + 3] = tmp.w;
            } else {
                for (int j = 0; j < 4; j++)
                    sB[row][col + j] = (globalRow < K && globalCol + j < N)
                                       ? B[globalRow * N + globalCol + j] : 0.0f;
            }
        }

        __syncthreads();

        // outer product + register accumulation
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            float regA[TM], regB[TN];
            for (int m = 0; m < TM; m++)
                regA[m] = sA[threadRow * TM + m][kk];
            for (int n = 0; n < TN; n++)
                regB[n] = sB[kk][threadCol * TN + n];
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    regC[m][n] += regA[m] * regB[n];
        }

        __syncthreads();
    }

    // float4 store
    for (int m = 0; m < TM; m++) {
        int globalRow = blockRowStart + threadRow * TM + m;
        for (int n = 0; n < TN; n += 4) {
            int globalCol = blockColStart + threadCol * TN + n;
            if (globalRow < M && globalCol + 3 < N) {
                float4 tmp = {regC[m][n], regC[m][n+1], regC[m][n+2], regC[m][n+3]};
                reinterpret_cast<float4*>(&C[globalRow * N + globalCol])[0] = tmp;
            } else {
                for (int j = 0; j < 4; j++)
                    if (globalRow < M && globalCol + j < N)
                        C[globalRow * N + globalCol + j] = regC[m][n + j];
            }
        }
    }
}

// ─── Helper: verify against cuBLAS ───────────────────────────────────────────
void verify(const float* C_ref, const float* C_out, int M, int N,
            const char* label) {
    double max_err = 0.0;
    for (int i = 0; i < M * N; i++) {
        double diff = fabs((double)C_ref[i] - (double)C_out[i]);
        if (diff > max_err) max_err = diff;
    }
    printf("[%s] max error vs cuBLAS: %.6e %s\n",
           label, max_err, max_err < 1e-2 ? "✓" : "✗ FAILED");
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {

    // 1. Matrix dimension setup (Default: 4096)
    int M = 4096, N = 4096, K = 4096;
    if (argc == 2) {
        M = N = K = atoi(argv[1]);
    } else if (argc == 4) {
        M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]);
    }

    // NOTE: srand(time(NULL)) is NOT used here to keep input data consistent
    // across different kernel versions for easier debugging and comparison.

    printf("Initializing GEMM Final: M=%d, N=%d, K=%d\n", M, N, K);

    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    // 2. Allocate host memory
    float *hA = (float*)malloc(sizeA);
    float *hB = (float*)malloc(sizeB);
    float *hC = (float*)malloc(sizeC);
    float *hC_ref = (float*)malloc(sizeC);

    // 3. Initialize with fixed seed
    srand(42);
    for (int i = 0; i < M * K; i++) hA[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) hB[i] = (float)rand() / RAND_MAX;

    // 4. Allocate device memory
    float *dA, *dB, *dC, *dC_ref;
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);
    cudaMalloc(&dC_ref, sizeC);

    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

    // 5. cuBLAS reference
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, dB, N, dA, K, &beta, dC_ref, N);
    cudaMemcpy(hC_ref, dC_ref, sizeC, cudaMemcpyDeviceToHost);

    // 6. Launch gemm_final
    dim3 block(64);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    // warmup
    gemm_final<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    // benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int RUNS = 10;
    cudaEventRecord(start);
    for (int i = 0; i < RUNS; i++)
        gemm_final<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    ms /= RUNS;
    printf("Execution Time: %.3f ms\n", ms);
    double flops  = 2.0 * M * N * K;
    double tflops = flops / (ms * 1e-3) / 1e12;
    printf("Throughput: %.2f TFLOPS\n", tflops);

    // 7. Verify
    cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);
    verify(hC_ref, hC, M, N, "gemm_final");

    // 8. Cleanup
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dC_ref);
    free(hA); free(hB); free(hC); free(hC_ref);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
