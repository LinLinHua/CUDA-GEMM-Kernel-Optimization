#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>


#define TILE_SIZE 16

__global__ void gemm_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    //global memory 
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N dimension

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    //move the data from global to shared memory
    for (int t = 0; t < numTiles; t++) {
        //shared memory
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        //calculate the result in this shared memory
        for (int k = 0; k < TILE_SIZE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }
    //move the data to global C
    if (row < M && col < N)
        C[row * N + col] = sum;
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

    printf("Initializing Tiled GEMM: M=%d, N=%d, K=%d\n", M, N, K);

    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    float *h_A   = (float*)malloc(sizeA);
    float *h_B   = (float*)malloc(sizeB);
    float *h_C   = (float*)malloc(sizeC);
    float *h_ref = (float*)malloc(sizeC);

    srand(42);
    for (size_t i = 0; i < (size_t)M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    //GPU need to annouce the block and grid dimension
    dim3 blockDim(16, 16);//on
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    // Warmup
    gemm_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int RUNS = 10;
    cudaEventRecord(start);
    for (int i = 0; i < RUNS; i++)
        gemm_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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
