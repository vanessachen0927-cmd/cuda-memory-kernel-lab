# Tesla T4 CUDA asynchronous pipeline — 2026-11-02

## Provenance and correctness

Results supplied from a Google Colab NVIDIA Tesla T4 GPU run, using experimental `src/stream_pipeline.cu`. This is the first reported run and has not been independently reproduced. Code compiled using CUDA `nvcc` after removing the legacy `cudaDeviceProp.deviceOverlap` field. CUDA compiler/driver details should be recorded in later runs.

```text
GPU=Tesla T4 N=8388608 iterations=20 chunks=16
asyncEngineCount=3
metric,value
single_stream_mean_ms,10.90844
single_stream_min_ms,10.53455
single_stream_max_ms,11.24458
four_stream_mean_ms,7.89712
four_stream_min_ms,7.71419
four_stream_max_ms,8.07004
single_over_four_speedup,1.38132
PASS: single-stream and four-stream GPU outputs match CPU reference
```

## Comparison

| Configuration | Mean latency (ms) | Min latency (ms) | Max latency (ms) |
| --- | ---: | ---: | ---: |
| One nonblocking CUDA stream | 10.90844 | 10.53455 | 11.24458 |
| Four nonblocking CUDA streams | 7.89712 | 7.71419 | 8.07004 |

The four-stream configuration shows **1.38132x throughput-oriented speedup** and **27.60% lower mean elapsed time** under this measurement setup.

Both use the same pinned allocations, input length, 16 chunks, vector-add kernel, output validation and 20 measured iterations. Timing is host wall-clock elapsed per iteration around enqueues plus `cudaDeviceSynchronize`, and excludes initial host/device allocation and data initialization. The pipeline has two H2D copies, one vector-add kernel and one D2H copy per chunk.

## Interpretation and limitations

- `asyncEngineCount=3` reports device hardware support for asynchronous copy engines. It does **not** by itself prove copy/compute overlap occurred.
- Multiple nonblocking streams allow independent chunks to overlap, subject to hardware resources and dependencies; this test does not include a profiler timeline, so **overlap is not verified**.
- Mean/min/max are calculated from iterations in one benchmark invocation. These are not independent-run variance statistics or confidence intervals.
- This benchmark's one-stream baseline uses 16 chunks and asynchronous operations in a *single* stream, not the original synchronous `cudaMemcpy` baseline, ensuring a more controlled stream-count comparison.
- Cloud load, GPU clocks, runtime setup, buffer sizes and stream scheduling may influence timings. Repeat whole processes and alternate configuration order to help assess bias before broad conclusions.

## Next steps

1. Independently repeat benchmark (and reverse measurement order) to assess stability.
2. Sweep number of chunks: 4, 8, 16, 32.
3. If Nsight Systems is available and permitted, capture a CUDA timeline to establish whether H2D, D2H and kernel periods actually overlap.
4. Profile or optimize tiled GEMM and compare correctness with a CPU/PyTorch reference.
