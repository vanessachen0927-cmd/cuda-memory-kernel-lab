# CUDA Memory & Kernel Performance Lab

An educational GPU performance experiment suite built for **real NVIDIA CUDA measurements** using **free GPU notebook runtimes**. It complements the separate [GPU PCIe Boot Co-Simulation](https://github.com/vanessachen0927-cmd/gpu-pcie-boot-simulator) project.

## Goals

- Explain and measure host↔device copies with pageable vs pinned memory.
- Implement and verify a real CUDA C++ kernel against a CPU reference.
- Compare CUDA block sizes (128/256/512) with CUDA-event timing.
- Report configuration, raw measurement values, methodology, and limitations.

## On macOS (no NVIDIA GPU required)

```bash
cmake -S . -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

CUDA targets are automatically omitted if no CUDA compiler is installed; the CPU test still runs.

## Free NVIDIA GPU notebook

1. Open [Google Colab](https://colab.research.google.com/) and request a free GPU runtime, if available. No payment is required for the free tier, but allocation is not guaranteed.
2. Upload or clone this project into Colab.
3. In a code cell, run:

```python
!nvidia-smi
!nvcc --version
!nvcc -std=c++17 -O2 -I src src/cuda_bench.cu -o cuda_bench
!./cuda_bench 1048576 50
```

If the GPU runtime lacks `nvcc`, use a compatible environment or install an available toolkit; do not invent CUDA results. Save the console output including GPU name, versions and input shape to `results/`.

## Metrics

- Host-wall time for **synchronous** pageable/pinned H2D and D2H `cudaMemcpy`; timings include runtime/API overhead. These are not pure PCIe bandwidth measurements.
- CUDA-event elapsed time for repeated vector-add kernels (five warmup launches) with block sizes 128/256/512.
- Nominal vector-add bandwidth estimate: `(2 input + 1 output) * N * sizeof(float) / kernel_time`. This is a simplified traffic estimate, not a profiler-measured DRAM read/write counter.
- GPU results always checked against the CPU reference.

## Caveats

**The asynchronous multi-stream pipeline is now implemented but has not yet been measured on a GPU. Shared-memory kernels, GEMM and NCCL remain outside the current implementation.** It is the reproducible baseline for those later experiments. Pinned allocation and pinned copies have costs; this first benchmark measures the copy operations after allocation, not full end-to-end application cost. Free Colab/Kaggle GPUs may vary between sessions and have time/resource limits. All reported results must come from an actual NVIDIA CUDA device.

## Roadmap

1. Baseline vector-add correctness, pageable/pinned copies, configurable kernel blocks (this version).
2. Repeat measurements with standard deviation, transfer-size sweeps, allocation-inclusive end-to-end benchmark.
3. Async `cudaMemcpyAsync`, multiple CUDA streams, compute/copy overlap and timeline plots.
4. GEMM tiled shared-memory kernels, Nsight profiling (only when platform permissions permit), and rigorous comparisons.

## Async CUDA stream pipeline (experimental)

The new `src/stream_pipeline.cu` performs chunked pinned-memory H2D copies, vector-add kernels and D2H copies, comparing one nonblocking stream against four. Both cases use the same input, same number of chunks and the same preallocated memory. All results are checked against the CPU reference. This is a deliberately simple pipeline, **not a guaranteed speedup or proof that operations overlap**.

On a CUDA-capable NVIDIA GPU:

```bash
nvcc -std=c++17 -O2 -I src src/stream_pipeline.cu -o stream_pipeline
./stream_pipeline 8388608 20 16
```

Outputs include GPU properties `asyncEngineCount`, `deviceOverlap`, host wall-clock per-iteration mean/min/max time, and single-stream divided by four-stream speedup. **No GPU timings for this new experiment are published yet.** Wall times include host enqueue and device synchronization, exclude initial memory allocation, and may vary under shared notebook GPU workloads. Record repeated independent runs, driver/toolkit versions and input shapes before drawing conclusions. See [baseline Tesla T4 sweep](results/tesla_t4_size_sweep_2026-11-02.md).
