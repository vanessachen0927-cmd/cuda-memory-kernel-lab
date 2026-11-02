# Tesla T4 CUDA baseline — November 2, 2026

## Provenance

**Source:** user-provided console output from `./cuda_bench 1048576 50` executed in a Google Colab GPU runtime. Measurements have **not** been independently repeated and are not a hardware-performance guarantee.

- GPU reported by the program: Tesla T4
- Input length: 1,048,576 `float` elements (4 MiB per buffer)
- Iterations: 50
- PyTorch CUDA version reported separately: 13.0 (this is not necessarily the toolkit/compiler version)
- `nvcc --version` ran successfully; full compiler version and driver versions not yet archived
- Output: `PASS: all CUDA outputs match CPU reference`

## Raw observed output

```text
GPU=Tesla T4 N=1048576 iterations=50
metric,value_ms
pageable_h2d,1.02126
pinned_h2d,0.48638
pageable_d2h,1.00373
pinned_d2h,0.35954
kernel_block128,0.05203
kernel_block128_nominal_gbps,241.84216
kernel_block256,0.05183
kernel_block256_nominal_gbps,242.75888
kernel_block512,0.05225
kernel_block512_nominal_gbps,240.80543
PASS: all CUDA outputs match CPU reference
```

## Calculated comparisons

| Operation | Pageable (ms) | Pinned (ms) | Pageable / pinned |
| --- | ---: | ---: | ---: |
| Host to Device | 1.02126 | 0.48638 | 2.10x |
| Device to Host | 1.00373 | 0.35954 | 2.79x |

The block-256 vector-add result (0.05183 ms) is the lowest in this **single run**, only about 0.38% below block-128 (0.05203 ms). This difference is too small to establish a reliable winner without independent repeated runs and variability reporting.

## Interpretation and limitations

- Copy timings are CPU host-wall averages of **synchronous** `cudaMemcpy` operations. They include runtime/API overhead and should not be called a measurement of PCIe link bandwidth.
- Pinned allocations and pageable/pinned buffer setup costs were **not included**. End-to-end application behavior can differ.
- Kernel latency is an average derived from a CUDA-event interval around 50 launches, after five warm-up launches.
- Nominal GB/s uses an **assumed** 12 bytes per vector-add element (two reads and one write), not observed DRAM traffic. Caching, transaction overhead and thermal/power variability may change results.
- Only one GPU instance, input size and iteration count were recorded. No confidence intervals or independent-run distribution were measured.
- Free cloud runtime hardware, contention, driver and toolkit versions can vary across sessions.

## Next controlled experiments

1. Repeat runs independently; preserve raw measurements and report median, standard deviation and range.
2. Sweep array sizes from 64 KiB to 64 MiB, accounting for the program's memory footprint.
3. Include pinned-allocation cost in an end-to-end comparison.
4. Study `cudaMemcpyAsync` with pinned host memory and CUDA streams, comparing serial execution with verified overlap.
