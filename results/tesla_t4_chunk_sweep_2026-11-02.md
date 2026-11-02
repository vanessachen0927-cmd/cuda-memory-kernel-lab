# Tesla T4 — CUDA Streams Chunk Sweep (2026-11-02)

## Provenance
Observed Google Colab Tesla T4 output supplied by project owner. `./stream_pipeline 8388608 20 CHUNKS` with `CHUNKS` set to 4, 8, 16, and 32. All four runs reported **PASS: single-stream and four-stream GPU outputs match CPU reference**. `asyncEngineCount=3`. No profiler timeline was captured.

## Raw measurements

| Chunks | Single mean (ms) | Single min (ms) | Single max (ms) | Four mean (ms) | Four min (ms) | Four max (ms) | Single / four speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 10.75370 | 10.55705 | 11.20554 | 8.18044 | 7.98971 | 8.45247 | 1.31456x |
| 8 | 10.70916 | 10.49183 | 11.02989 | 7.86528 | 7.71633 | 8.09940 | 1.36157x |
| 16 | 10.91821 | 10.60926 | 11.31570 | 7.80788 | 7.53866 | 8.08133 | 1.39836x |
| 32 | 11.46204 | 11.22142 | 11.70254 | 8.34898 | 8.07859 | 8.61869 | 1.37287x |

## Interpretation

- **Four streams were faster than one stream** at each chunk count in these observations.
- **16 chunks yielded the lowest observed four-stream mean latency**, 7.80788 ms. Compared with its one-stream baseline at the same chunk count, the mean was ~28.49% lower and the ratio was 1.39836x.
- The 16-chunk result improves upon the 8-chunk four-stream mean by only 0.05740 ms (~0.73%), which is insufficient to establish a general performance winner without independent repeats and run-order control.
- At 32 chunks, the four-stream mean increases to 8.34898 ms, consistent with possible per-chunk overhead, but this cause is not verified.
- `asyncEngineCount=3` supports the *possibility* of hardware copy-engine concurrency, **not proof of transfer/compute overlap**.

## Timing and limitations

- One input shape (`N=8388608` floats), one GPU instance, 20 timed iterations per config, and one invocation per chunk count.
- Timings represent host-wall elapsed time around asynchronous enqueuing and `cudaDeviceSynchronize`; initial allocations and buffer initialization are excluded.
- The code measures the single-stream case **before** the four-stream case, so order-dependent effects may bias comparisons.
- Min/max within a process are not confidence intervals and do not replace inter-run variability measurements.
- Driver/toolkit versions and GPU clocks/load were not archived in this sweep. No Nsight Systems timeline was recorded.
- This benchmark is not evidence of an optimal chunk count for other GPU architectures or production workloads.

## Next steps
1. Repeat each configuration independently, shuffle execution order and report median and variability.
2. Try smaller and larger N to test whether chunk-size trends persist.
3. If supported by the free runtime, capture Nsight Systems timelines to inspect H2D, kernel, and D2H overlap.
4. Proceed to tiled GEMM with correctness tests against a reference implementation.
