# Performance evidence

**Status:** historical pre-0.2 baseline; superseded by the core certification  
**Date:** 2026-07-19  
**Commit:** `04297ebdb05df853a05003232d621a2d7acbda10`  
**Working tree during run:** dirty  
**Host:** AMD Ryzen 5 5500, 12 logical CPUs, 33.4 GB physical memory  
**Python:** 3.13.13  

This report records the original 0.1 implementation before bounded worker execution. It is kept
as a regression baseline and must not be presented as the current certification. It records one
local execution of the methodology in
[Performance and load testing](performance.md). Because the working tree contained the benchmark
implementation itself, rerun it from a clean committed tree before quoting these numbers in a CV,
release note, or public performance claim.

## Commands

```fish
.venv/bin/python docs/benchmarks/performance.py \
    --jobs 20 \
    --samples 100 \
    --warmups 3 \
    --fragments 3 \
    --resource-jobs 1000 \
    --concurrency 1,4,16 \
    --workloads map,nested,full \
    --adapter-delay-ms 1 \
    --output /tmp/promptnest-performance-100.json

.venv/bin/python docs/benchmarks/resilience.py \
    --output /tmp/promptnest-resilience-100.json
```

## Results

All latency percentiles below use 100 complete batch samples and R-7 interpolation.

| Workload | Concurrency | Calls/batch | Latency p95 | Throughput p50 | Speedup |
|---|---:|---:|---:|---:|---:|
| map | 1 | 20 | 22.414 ms | 905.642 jobs/s | 1.000× |
| map | 4 | 20 | 5.988 ms | 3,408.074 jobs/s | 3.763× |
| map | 16 | 20 | 2.682 ms | 7,862.794 jobs/s | 8.682× |
| nested | 1 | 80 | 88.976 ms | 227.666 jobs/s | 1.000× |
| nested | 4 | 80 | 24.145 ms | 863.071 jobs/s | 3.791× |
| nested | 16 | 80 | 7.120 ms | 2,978.830 jobs/s | 13.084× |
| full | 1 | 81 | 90.652 ms | 224.103 jobs/s | 1.000× |
| full | 4 | 81 | 25.322 ms | 819.114 jobs/s | 3.655× |
| full | 16 | 81 | 8.156 ms | 2,518.331 jobs/s | 11.237× |

For the independent map workload, p95 orchestration overhead
(`observed − theoretical adapter service time`) was:

- concurrency 1: 2.414 ms per 20-job batch;
- concurrency 4: 0.988 ms per 20-job batch;
- concurrency 16: 0.682 ms per 20-job batch.

The separate 1,000-job resource probes reported approximately 4.4–4.7 MB peak traced Python
memory for map and 9.0–9.2 MB for nested/full. Pending tasks peaked at 2,001 for map and 4,001 for
nested/full. This is evidence that active adapter calls are bounded while pending work is not;
it is not evidence of complete backpressure.

## Failure behavior

All deterministic 100-job resilience scenarios passed:

- 100 first-attempt failures recovered after exactly 100 retries;
- timeout cleanup left zero active adapter calls;
- 50 successful jobs were retained when the other 50 failed;
- caller cancellation left zero active calls and cancelled the eight admitted calls.

## Claim boundary

This execution supports claims about deterministic PromptNest orchestration overhead, throughput
scaling, complete map/nested/reduce coverage, resource growth, and cleanup under synthetic
failures. It does not support TTFT, real-provider latency, model latency, complete backpressure, or
an independent certification claim.
