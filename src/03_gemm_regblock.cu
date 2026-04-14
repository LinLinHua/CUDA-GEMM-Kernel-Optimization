#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BM 64   // block tile size in M dimension
#define BN 64   // block tile size in N dimension
#define BK 16   // block tile size in K dimension
#define TM 8    // each thread computes TM output rows
#define TN 8    // each thread computes TN output cols

// totalThreads = (BM/TM) * (BN/TN) = 8 * 8 = 64

__global__ void gemm_regblock(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    int threadRow = threadIdx.x / (BN / TN);
    int threadCol = threadIdx.x % (BN / TN);

    int blockRowStart = blockIdx.y * BM;
    int blockColStart = blockIdx.x * BN;

    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    float regC[TM][TN] = {0.0f};

    int totalThreads = (BM / TM) * (BN / TN);  // 64
    int tid = threadIdx.x;

    for (int k = 0; k < K; k += BK) {

        // Load A tile
        for (int i = tid; i < BM * BK; i += totalThreads) {
            int r = i / BK, c = i % BK;
            // FIX: was A[r * K + k + c] — missing blockRowStart offset,
            //      causing all blocks except blockIdx.y==0 to read wrong rows.
            sA[r][c] = (blockRowStart + r < M && k + c < K)
                       ? A[(blockRowStart + r) * K + k + c] : 0.0f;
        }

        // Load B tile
        for (int i = tid; i < BK * BN; i += totalThreads) {
            int r = i / BN, c = i % BN;
            // FIX: was B[(k + r) * N + c] — missing blockColStart offset,
            //      causing all blocks except blockIdx.x==0 to read wrong cols.
            sB[r][c] = (k + r < K && blockColStart + c < N)
                       ? B[(k + r) * N + blockColStart + c] : 0.0f;
        }

        __syncthreads();

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

    // Write results back to global memory
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
            int globalRow = blockRowStart + threadRow * TM + m;
            int globalCol = blockColStart + threadCol * TN + n;
            // FIX: was C[row * N + col] — missing block offsets,
            //      all blocks wrote to the same top-left region of C.
            if (globalRow < M && globalCol < N)
                C[globalRow * N + globalCol] = regC[m][n];
        }
    }
}

void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

bool verify(const float* ref, const float* out, int size) {
    for (int i = 0; i < size; i++)
        if (fabsf(ref[i] - out[i]) > 1e-3f) {
            printf("MISMATCH at %d: ref=%.4f, got=%.4f\n", i, ref[i], out[i]);
            return false;
        }
    return true;
}

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc == 2) {
        M = N = K = atoi(argv[1]);
    } else if (argc == 4) {
        M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]);
    }

    printf("Initializing RegBlock GEMM: M=%d, N=%d, K=%d\n", M, N, K);

    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    float* h_A   = (float*)malloc(sizeA);
    float* h_B   = (float*)malloc(sizeB);
    float* h_C   = (float*)malloc(sizeC);
    float* h_ref = (float*)malloc(sizeC);

    srand(42);
    for (size_t i = 0; i < (size_t)M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(64);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    // Warmup
    gemm_regblock<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int RUNS = 10;
    cudaEventRecord(start);
    for (int i = 0; i < RUNS; i++)
        gemm_regblock<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= RUNS;
    printf("Execution Time: %.3f ms\n", ms);

    double tflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e12;
    printf("Throughput: %.2f TFLOPS\n", tflops);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    if (M <= 512) {
        gemm_cpu(h_A, h_B, h_ref, M, N, K);
        bool ok = verify(h_ref, h_C, M * N);
        printf("Result: %s\n", ok ? "CORRECT" : "WRONG");
    } else {
        printf("Result: skipped (matrix too large for CPU verify)\n");
    }

    free(h_A); free(h_B); free(h_C); free(h_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
