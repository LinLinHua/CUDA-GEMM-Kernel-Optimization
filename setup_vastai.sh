#!/bin/bash

set -e  # Exit on any error

echo "======================================================"
echo " GEMM Project Setup – vast.ai"
echo "======================================================"

# ── 1. System packages ────────────────────────────────────────────────────
echo "[1/5] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    build-essential \
    git \
    wget \
    curl \
    vim \
    python3-pip \
    numactl

# ── 2. Verify CUDA ────────────────────────────────────────────────────────
echo "[2/5] Checking CUDA environment..."
nvidia-smi
nvcc --version
echo "CUDA library path: $(ldconfig -p | grep libcuda | head -1)"

# Ensure nvcc is on PATH (some images need this)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# ── 3. Verify NCU (Nsight Compute) ───────────────────────────────────────
# On vast.ai you have root access, so NCU works without --root-override.
echo "[3/5] Checking Nsight Compute..."
if command -v ncu &> /dev/null; then
    echo "ncu found: $(ncu --version | head -1)"
else
    echo "WARNING: ncu not found. Install Nsight Compute separately if needed."
    echo "Download: https://developer.nvidia.com/nsight-compute"
fi

# ── 4. Python packages (for future PyTorch MHA baseline) ─────────────────
echo "[4/5] Installing Python packages..."
pip install --quiet \
    torch \
    numpy \
    matplotlib \
    jupyterlab

# ── 5. Build the GEMM project ────────────────────────────────────────────
echo "[5/5] Building GEMM project..."
if [ -d ~/00_GEMM_project ]; then
    cd ~/00_GEMM_project
    make clean && make all
    echo ""
    echo "Build complete. Run benchmark with:"
    echo "  bash benchmark.sh | tee results/final_performance_table.md"
else
    echo "WARNING: ~/00_GEMM_project not found."
    echo "Upload the project first:  scp -P <port> -r ./00_GEMM_project root@<host>:~/"
fi

echo ""
echo "======================================================"
echo " Setup complete."
echo ""
echo " Key commands:"
echo "   make all                   – build all kernels"
echo "   bash benchmark.sh          – run full benchmark"
echo "   make profile               – NCU roofline (fast, ~5 passes)"
echo "   make profile_full          – NCU full set  (slow, ~49 passes)"
echo "   make profile_bm128         – NCU roofline for BM128 tuning variant"
echo ""
echo " Start JupyterLab (optional):"
echo "   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo "   Then open: http://localhost:8888 (via SSH tunnel -L 8888:localhost:8888)"
echo "======================================================"
