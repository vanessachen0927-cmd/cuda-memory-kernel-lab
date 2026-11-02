# Tesla T4 — Six-size CUDA transfer and vector-add sweep

**Date:** 2026-11-02. **Source:** console output supplied from a Google Colab Tesla T4 runtime. These measurements are independently **unverified** by the repository's CI because CI only has a CPU. Every GPU run reported `PASS: all CUDA outputs match CPU reference`.

**Command:** `./cuda_bench N 50` for six values of `N`. The transfer sizes below are bytes per single `float32` buffer. See [raw sweep CSV](tesla_t4_size_sweep_2026-11-02.csv) and [first 4 MiB baseline](tesla_t4_baseline_2026-11-02.md).

| N elements | Buffer | Pageable H2D ms | Pinned H2D ms | H2D speedup | Pageable D2H ms | Pinned D2H ms | D2H speedup | Fastest block (single run) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16,384 | 64 KiB | 0.04092 | 0.01941 | **2.11x** | 0.03016 | 0.01574 | **1.92x** | 128 |
| 65,536 | 256 KiB | 0.09136 | 0.04049 | **2.26x** | 0.08633 | 0.03043 | **2.84x** | 256 |
| 262,144 | 1 MiB | 0.33148 | 0.12817 | **2.59x** | 0.31694 | 0.09685 | **3.27x** | 128 |
| 1,048,576 | 4 MiB | 1.00658 | 0.50932 | **1.98x** | 1.06444 | 0.35704 | **2.98x** | 256 |
| 4,194,304 | 16 MiB | 3.50603 | 1.94744 | **1.80x** | 3.66867 | 1.40739 | **2.61x** | 256 |
| 8,388,608 | 32 MiB | 6.97709 | 3.97426 | **1.76x** | 7.37586 | 2.81107 | **2.62x** | 256 |

## What the data show

- Pinned synchronous copies were faster than pageable synchronous copies at every measured size for both directions in this Colab session.
- H2D relative improvements varied from **1.76x** to **2.59x**; D2H from **1.92x** to **3.27x**.
- For the three largest sizes, block-128, block-256 and block-512 kernel times were close. This sweep is insufficient to conclude a consistently best block size.
- The calculated nominal bandwidth for the 1 MiB test exceeded 600 GB/s. This is **not observed DRAM bandwidth**; the formula assumes 3 float32 accesses per element, which may differ from actual DRAM traffic because of caches and other effects.
- Kernel latency for 64 KiB was slightly larger than for 256 KiB, illustrating how short measurements can be sensitive to noise.

## Measurement limitations

- `time_copy_ms` uses host wall-clock timing around **synchronous** `cudaMemcpy`. It includes API/runtime overhead, and does not isolate physical PCIe data transfer.
- Copy measurements exclude pinned allocation/deallocation and pageable staging/setup expenses.
- Kernel timing is a CUDA-event elapsed interval divided by 50 launches after warmups. A reported value is one benchmark invocation, not an independent-run distribution.
- Input contents, concurrent GPU users, power/clock state, driver and CUDA toolchain versions were not all archived. No standard deviation, confidence interval or statistical significance claim is possible.
- Do not extrapolate these Tesla T4 numbers to Ampere, Hopper, Blackwell, or to actual GPU boot latency.

## Follow-up

1. Repeat the full sweep multiple times and archive GPU info and raw run outputs.
2. Include allocation/setup costs to compare end-to-end pageable vs pinned workflow.
3. Add `cudaMemcpyAsync`, pinned buffers, two streams and explicit correctness tests; compare serialized vs overlapped workload.
4. Profile the kernels when available, then examine coalescing and shared-memory GEMM.
