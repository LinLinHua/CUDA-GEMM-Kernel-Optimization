#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// =========================================================================
// 04_gemm_profile.cu
// ------------------
// Stripped-down single-run version of 04_gemm_float4 for NCU profiling.
// Single launch = clean profile without loop overhead in the trace.
// Usage: ncu --set roofline -o results/04_profile ./04_gemm_profile 4096
// =========================================================================

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

// totalThreads = (BM/TM) * (BN/TN) = 16 * 16 = 256

__global__ void gemm_float4(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    int threadRow = threadIdx.x / (BN / TN);
    int threadCol = threadIdx.x % (BN / TN);

    int blockRowStart = blockIdx.y * BM;
    int blockColStart = blockIdx.x * BN;

    // +4 padding to eliminate bank conflicts (see 04_gemm_float4.cu for explanation)
    __shared__ float sA[BM][BK + 4];
    __shared__ float sB[BK][BN + 4];

    float regC[TM][TN] = {0.0f};

    int totalThreads = (BM / TM) * (BN / TN);  // 256
    int tid = threadIdx.x;

    for (int k = 0; k < K; k += BK) {

        for (int i = tid; i < (BM * BK) / 4; i += totalThreads) {
            int row      = i / (BK / 4);
            int col      = (i % (BK / 4)) * 4;
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

        for (int i = tid; i < (BK * BN) / 4; i += totalThreads) {
            int row      = i / (BN / 4);
            int col      = (i % (BN / 4)) * 4;
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

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc == 2) {
        M = N = K = atoi(argv[1]);
    } else if (argc == 4) {
        M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]);
    }

    printf("Initializing Float4 Profile: M=%d, N=%d, K=%d\n", M, N, K);

    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);

    srand(42);
    for (size_t i = 0; i < (size_t)M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(256);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    // Warmup (excluded from NCU by --kernel-name filter)
    gemm_float4<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Single timed run for profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gemm_float4<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Execution Time: %.3f ms\n", ms);
    printf("Throughput: %.2f TFLOPS\n", (2.0 * M * N * K) / (ms * 1e-3) / 1e12);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
