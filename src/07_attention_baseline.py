import torch
import torch.nn.functional as F
import math
import time

# =========================================================================
# 07_attention_baseline.py
# -------------------------------------------------------------------------
# Three attention implementations for comparison:
#   1. Naive   : explicit QK^T, softmax, x V  (materialized N x N matrix)
#   2. PyTorch : F.scaled_dot_product_attention (fused, FlashAttention-style)
#   3. Reference for CUDA kernel correctness verification
# =========================================================================

def attention_naive(Q, K, V):
    """
    Naive attention: materializes full N x N score matrix to HBM.
    O = softmax(QK^T / sqrt(d)) @ V
    HBM traffic: O(N^2) — this is what the fused kernel eliminates.
    """
    d = Q.shape[-1]
    S = Q @ K.transpose(-2, -1) / math.sqrt(d)   # (B, H, N, N)
    P = torch.softmax(S, dim=-1)                   # (B, H, N, N)
    O = P @ V                                       # (B, H, N, d)
    return O

def attention_pytorch(Q, K, V):
    """
    PyTorch SDPA: fused FlashAttention-style kernel.
    Used as the performance target for our CUDA implementation.
    """
    return F.scaled_dot_product_attention(Q, K, V)

def benchmark(fn, *args, runs=20, warmup=5, label=""):
    # Warmup
    for _ in range(warmup):
        out = fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(runs):
        out = fn(*args)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / runs
    # FLOPs: 2 * B * H * N * N * d (QK^T) + 2 * B * H * N * N * d (PV)
    B, H, N, d = args[0].shape
    flops  = 4.0 * B * H * N * N * d
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"{label:30s}  {ms:.3f} ms  {tflops:.2f} TFLOPS")
    return out, ms

def verify(ref, out, label):
    max_err = (ref - out).abs().max().item()
    status  = "✓" if max_err < 1e-2 else "✗ FAILED"
    print(f"  [{label}] max error vs naive: {max_err:.6e} {status}")

# ── Config ────────────────────────────────────────────────────────────────
device = torch.device("cuda")
dtype  = torch.float16    # FP16 — matches A100 Tensor Core target

configs = [
    # (B,  H,   N,    d)   B=batch, H=heads, N=seq_len, d=head_dim
    (1,   8,  1024,  64),
    (1,   8,  2048,  64),
    (1,   8,  4096,  64),
    (1,  16,  4096,  64),
]

print(f"\n{'Config':30s}  {'Time':>10}  {'TFLOPS':>10}")
print("-" * 60)

for (B, H, N, d) in configs:
    Q = torch.randn(B, H, N, d, device=device, dtype=dtype)
    K = torch.randn(B, H, N, d, device=device, dtype=dtype)
    V = torch.randn(B, H, N, d, device=device, dtype=dtype)

    label = f"B={B} H={H} N={N} d={d}"
    print(f"\n{label}")

    ref, _  = benchmark(attention_naive,   Q, K, V, label="  naive")
    out, _  = benchmark(attention_pytorch, Q, K, V, label="  pytorch SDPA")

    verify(ref.float(), out.float(), "pytorch SDPA")


    '''
    Config                                Time      TFLOPS
------------------------------------------------------------
B=1 H=8 N=1024 d=64
  naive                         0.121 ms  17.68 TFLOPS
  pytorch SDPA                  0.051 ms  41.90 TFLOPS
  [pytorch SDPA] max error vs naive: 6.103516e-04 ✓
B=1 H=8 N=2048 d=64
  naive                         0.505 ms  17.00 TFLOPS
  pytorch SDPA                  0.111 ms  77.10 TFLOPS
  [pytorch SDPA] max error vs naive: 4.882812e-04 ✓
B=1 H=8 N=4096 d=64
  naive                         1.897 ms  18.11 TFLOPS
  pytorch SDPA                  0.318 ms  108.15 TFLOPS
  [pytorch SDPA] max error vs naive: 2.441406e-04 ✓
B=1 H=16 N=4096 d=64
  naive                         3.736 ms  18.39 TFLOPS
  pytorch SDPA                  0.516 ms  133.17 TFLOPS
  [pytorch SDPA] max error vs naive: 2.441406e-04 ✓
    '''