#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

// =========================================================================
// 08_fused_attention.cu  —  Fused Multi-Head Attention
// -------------------------------------------------------------------------
// Extended from 06_gemm_float4_ultimate.cu.
//
// Thread layout:
//   Each block handles one Q tile (Br query rows).
//   64 threads split into 8 row-groups × 8 col-groups.
//   Each thread owns TM=8 query rows × TN=4 KV cols of the score tile.
//
//   For softmax, threads in the same row-group (same threadRow)
//   need to reduce across all Bc=32 cols → 8 threads per row,
//   each with TN=4 cols → warp shuffle reduces to get full row max/sum.
//
// Shared memory budget (48KB limit on A100):
//   sQ[Br][D+4]  = 64×68×4  = 17408 bytes
//   sK[Bc][D+4]  = 32×68×4  =  8704 bytes
//   sV[Bc][D+4]  = 32×68×4  =  8704 bytes
//   total = 34816 bytes = 34KB  ✓
// =========================================================================

#define Br  64    // query tile rows
#define Bc  32    // KV tile cols
#define D   64    // head dimension
#define TM  8     // reg rows per thread
#define TN  4     // reg cols per thread
// threads/block = (Br/TM) * (Bc/TN) = 8 * 8 = 64

// ── Warp reduce: max across all threads in warp ───────────────────────────
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// ── Warp reduce: sum across all threads in warp ───────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void __launch_bounds__(64, 2)
fused_attention(
    const float* __restrict__ Q,   // (H, N, D)
    const float* __restrict__ K,   // (H, N, D)
    const float* __restrict__ V,   // (H, N, D)
    float*       __restrict__ O,   // (H, N, D)
    int N)
{
    int head      = blockIdx.x;
    int q_tile    = blockIdx.y;
    int tid       = threadIdx.x;

    // Thread position within the Br×Bc score tile
    // threadRow: which group of TM query rows (0..7)
    // threadCol: which group of TN KV cols   (0..7)
    int threadRow = tid / (Bc / TN);   // 0..7
    int threadCol = tid % (Bc / TN);   // 0..7

    int blockRowStart = q_tile * Br;
    int totalThreads  = (Br / TM) * (Bc / TN);  // 64

    const float* Qh = Q + head * N * D;
    const float* Kh = K + head * N * D;
    const float* Vh = V + head * N * D;
    float*       Oh = O + head * N * D;

    // ── Shared memory ─────────────────────────────────────────────────
    __shared__ float sQ[Br][D + 4];
    __shared__ float sK[Bc][D + 4];
    __shared__ float sV[Bc][D + 4];

    // ── Load Q tile into SRAM (stays for all KV tiles) ────────────────
    for (int i = tid; i < (Br * D) / 4; i += totalThreads) {
        int row      = i / (D / 4);
        int col      = (i % (D / 4)) * 4;
        int globalRow = blockRowStart + row;
        if (globalRow < N) {
            float4 tmp = reinterpret_cast<const float4*>(
                             &Qh[globalRow * D + col])[0];
            sQ[row][col]     = tmp.x;
            sQ[row][col + 1] = tmp.y;
            sQ[row][col + 2] = tmp.z;
            sQ[row][col + 3] = tmp.w;
        } else {
            sQ[row][col] = sQ[row][col+1] =
            sQ[row][col+2] = sQ[row][col+3] = 0.0f;
        }
    }
    __syncthreads();

    // ── Per-thread accumulators ────────────────────────────────────────
    // regO[TM][TN]: output accumulator for this thread's tile
    // m[TM], l[TM]: running max and sum, one per query row
    float regO[TM][TN] = {};
    float m[TM], l[TM];
    #pragma unroll
    for (int i = 0; i < TM; i++) { m[i] = -FLT_MAX; l[i] = 0.0f; }

    const float scale = 1.0f / sqrtf((float)D);
    int num_kv_tiles  = (N + Bc - 1) / Bc;

    for (int kv = 0; kv < num_kv_tiles; kv++) {

        // ── Load K tile ───────────────────────────────────────────────
        for (int i = tid; i < (Bc * D) / 4; i += totalThreads) {
            int row      = i / (D / 4);
            int col      = (i % (D / 4)) * 4;
            int globalRow = kv * Bc + row;
            if (globalRow < N) {
                float4 tmp = reinterpret_cast<const float4*>(
                                 &Kh[globalRow * D + col])[0];
                sK[row][col]     = tmp.x;
                sK[row][col + 1] = tmp.y;
                sK[row][col + 2] = tmp.z;
                sK[row][col + 3] = tmp.w;
            } else {
                sK[row][col] = sK[row][col+1] =
                sK[row][col+2] = sK[row][col+3] = 0.0f;
            }
        }

        // ── Load V tile ───────────────────────────────────────────────
        for (int i = tid; i < (Bc * D) / 4; i += totalThreads) {
            int row      = i / (D / 4);
            int col      = (i % (D / 4)) * 4;
            int globalRow = kv * Bc + row;
            if (globalRow < N) {
                float4 tmp = reinterpret_cast<const float4*>(
                                 &Vh[globalRow * D + col])[0];
                sV[row][col]     = tmp.x;
                sV[row][col + 1] = tmp.y;
                sV[row][col + 2] = tmp.z;
                sV[row][col + 3] = tmp.w;
            } else {
                sV[row][col] = sV[row][col+1] =
                sV[row][col+2] = sV[row][col+3] = 0.0f;
            }
        }
        __syncthreads();

        // ── GEMM-1: regS = sQ_tile @ sK_tile^T ───────────────────────
        // Each thread computes TM×TN scores
        float regS[TM][TN] = {};
        #pragma unroll
        for (int d = 0; d < D; d++) {
            float regA[TM], regB[TN];
            #pragma unroll
            for (int m_ = 0; m_ < TM; m_++)
                regA[m_] = sQ[threadRow * TM + m_][d];
            #pragma unroll
            for (int n = 0; n < TN; n++)
                regB[n] = sK[threadCol * TN + n][d];
            #pragma unroll
            for (int m_ = 0; m_ < TM; m_++)
                for (int n = 0; n < TN; n++)
                    regS[m_][n] += regA[m_] * regB[n];
        }

        // Scale and mask
        #pragma unroll
        for (int m_ = 0; m_ < TM; m_++) {
            int q_global = blockRowStart + threadRow * TM + m_;
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                regS[m_][n] *= scale;
                int kv_global = kv * Bc + threadCol * TN + n;
                if (q_global >= N || kv_global >= N)
                    regS[m_][n] = -FLT_MAX;
            }
        }

        // ── Online softmax with warp reduction ────────────────────────
        // Each thread has TN=4 cols of one row.
        // Threads with the same threadRow share the same query row.
        // There are Bc/TN = 8 such threads per row → fits in half a warp.
        // We use warp shuffle to reduce max and sum across these 8 threads.
        //
        // Warp layout (64 threads):
        //   thread  0.. 7: threadRow=0, threadCol=0..7
        //   thread  8..15: threadRow=1, threadCol=0..7
        //   ...
        //   thread 56..63: threadRow=7, threadCol=0..7
        //
        // Each group of 8 threads (same threadRow) is contiguous in the warp
        // and can use __shfl_down_sync with mask 0xff shifted to their lane.

        float regP[TM][TN];

        #pragma unroll
        for (int m_ = 0; m_ < TM; m_++) {

            // Step 1: find local max over this thread's TN cols
            float local_max = -FLT_MAX;
            #pragma unroll
            for (int n = 0; n < TN; n++)
                local_max = fmaxf(local_max, regS[m_][n]);

            // Step 2: reduce max across 8 threads in same row-group
            // These 8 threads are at lanes (threadRow*8 + 0..7) within warp
            // Use mask for just these 8 threads
            int row_base = threadRow * (Bc / TN);  // base lane of row-group
            unsigned mask = 0xff << (row_base % 32);  // 8-thread mask

            // Reduce within the 8-thread group
            #pragma unroll
            for (int offset = 4; offset > 0; offset >>= 1)
                local_max = fmaxf(local_max,
                    __shfl_down_sync(mask, local_max, offset));

            // Broadcast full-row max from lane 0 of group to all 8
            float row_max = __shfl_sync(mask, local_max, row_base % 32);

            // Step 3: online softmax update
            float m_new = fmaxf(m[m_], row_max);
            float alpha = expf(m[m_] - m_new);

            // Step 4: compute p and local sum
            float local_sum = 0.0f;
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                regP[m_][n] = expf(regS[m_][n] - m_new);
                local_sum  += regP[m_][n];
            }

            // Step 5: reduce sum across 8 threads in same row-group
            #pragma unroll
            for (int offset = 4; offset > 0; offset >>= 1)
                local_sum += __shfl_down_sync(mask, local_sum, offset);
            float row_sum = __shfl_sync(mask, local_sum, row_base % 32);

            // Step 6: update running stats and rescale output
            l[m_] = l[m_] * alpha + row_sum;
            #pragma unroll
            for (int n = 0; n < TN; n++)
                regO[m_][n] *= alpha;
            m[m_] = m_new;
        }

        // ── GEMM-2: regO += regP @ sV ─────────────────────────────────
        // regP[TM][TN]: this thread's slice of the Br×Bc attention weights
        // sV[Bc][D]:    KV tile in SRAM
        // Each thread accumulates its TM rows × TN cols of output O
        //
        // regP[m_][n] is the weight for (query row m_, KV col threadCol*TN+n)
        // Contribution to output col d:
        //   regO[m_][d_local] += regP[m_][n] * sV[threadCol*TN + n][d]
        //
        // But each thread only owns TN output cols (threadCol*TN .. +TN).
        // We accumulate the full D=64 output by iterating over all Bc KV cols,
        // using regP values from all threads in the row-group via shfl.

        #pragma unroll
        for (int m_ = 0; m_ < TM; m_++) {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                // KV index this (m_, n) weight corresponds to
                int kv_col = threadCol * TN + n;
                // Accumulate into this thread's TN output cols
                #pragma unroll
                for (int d = 0; d < TN; d++) {
                    int out_col = threadCol * TN + d;
                    regO[m_][d] += regP[m_][n] * sV[kv_col][out_col];
                }
            }
        }

        __syncthreads();
    }

    // ── Normalize and write O ──────────────────────────────────────────
    #pragma unroll
    for (int m_ = 0; m_ < TM; m_++) {
        int globalRow = blockRowStart + threadRow * TM + m_;
        if (globalRow < N) {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int globalCol = threadCol * TN + n;
                Oh[globalRow * D + globalCol] = regO[m_][n] / l[m_];
            }
        }
    }
}

// ── CPU reference ─────────────────────────────────────────────────────────
void attention_cpu(const float* Q, const float* K, const float* V,
                   float* O, int H, int N) {
    const float scale = 1.0f / sqrtf((float)D);
    for (int h = 0; h < H; h++) {
        const float* Qh = Q + h * N * D;
        const float* Kh = K + h * N * D;
        const float* Vh = V + h * N * D;
        float*       Oh = O + h * N * D;
        for (int i = 0; i < N; i++) {
            float mx = -FLT_MAX, sum = 0.0f;
            float s[4096];
            for (int j = 0; j < N; j++) {
                float dot = 0.0f;
                for (int k = 0; k < D; k++)
                    dot += Qh[i*D+k] * Kh[j*D+k];
                s[j] = dot * scale;
                mx = fmaxf(mx, s[j]);
            }
            for (int j = 0; j < N; j++) {
                s[j] = expf(s[j] - mx);
                sum += s[j];
            }
            for (int k = 0; k < D; k++) {
                float acc = 0.0f;
                for (int j = 0; j < N; j++)
                    acc += (s[j]/sum) * Vh[j*D+k];
                Oh[i*D+k] = acc;
            }
        }
    }
}

void verify(const float* ref, const float* out, int size, const char* label) {
    double max_err = 0.0;
    for (int i = 0; i < size; i++)
        max_err = fmax(max_err, fabs((double)ref[i] - (double)out[i]));
    printf("[%s] max error: %.2e %s\n",
           label, max_err, max_err < 1e-2 ? "✓" : "✗ FAILED");
}

int main(int argc, char** argv) {
    int N = 1024, H = 8;
    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) H = atoi(argv[2]);

    printf("Fused Attention: H=%d N=%d D=%d  (Br=%d Bc=%d TM=%d TN=%d)\n",
           H, N, D, Br, Bc, TM, TN);

    size_t sz = (size_t)H * N * D * sizeof(float);
    float *hQ=(float*)malloc(sz), *hK=(float*)malloc(sz);
    float *hV=(float*)malloc(sz), *hO=(float*)malloc(sz);
    float *hRef=(float*)malloc(sz);

    srand(42);
    for (size_t i = 0; i < (size_t)H*N*D; i++) {
        hQ[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
        hK[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
        hV[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
    }

    printf("Running CPU reference...\n");
    attention_cpu(hQ, hK, hV, hRef, H, N);

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, sz); cudaMalloc(&dK, sz);
    cudaMalloc(&dV, sz); cudaMalloc(&dO, sz);
    cudaMemcpy(dQ, hQ, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, sz, cudaMemcpyHostToDevice);

    dim3 grid(H, (N + Br - 1) / Br);
    dim3 block(64);

    fused_attention<<<grid, block>>>(dQ, dK, dV, dO, N);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int RUNS = 50;
    cudaEventRecord(start);
    for (int i = 0; i < RUNS; i++)
        fused_attention<<<grid, block>>>(dQ, dK, dV, dO, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= RUNS;

    double tflops = (4.0 * H * N * N * D) / (ms * 1e-3) / 1e12;
    printf("Execution Time: %.3f ms\n", ms);
    printf("Throughput: %.2f TFLOPS\n", tflops);

    cudaMemcpy(hO, dO, sz, cudaMemcpyDeviceToHost);
    verify(hRef, hO, H*N*D, "fused_attention");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    free(hQ); free(hK); free(hV); free(hO); free(hRef);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}