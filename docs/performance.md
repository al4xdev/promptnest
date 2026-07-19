# Performance and load testing

PromptNest's natural performance metrics are completed jobs per second and batch completion
latency under different concurrency limits. This repository includes a deterministic benchmark
that reports throughput and latency distributions without making network or model-provider calls.

The benchmark is an orchestration benchmark, not evidence about the latency of OpenAI, LangChain,
LangGraph, CrewAI, a particular model, or a production network.

## Reproduce the benchmark

From the repository root:

```fish
uv sync --dev
uv run python docs/benchmarks/performance.py \
    --jobs 100 \
    --samples 100 \
    --warmups 3 \
    --concurrency 1,4,16 \
    --workloads map,nested,full \
    --fragments 3 \
    --resource-jobs 1000 \
    --adapter-delay-ms 2 \
    --output /tmp/promptnest-performance.json

uv run python docs/benchmarks/resilience.py \
    --output /tmp/promptnest-resilience.json
```

The command writes the same JSON it prints. `/tmp` is intentional: benchmark results depend on
the machine and should not silently become repository claims. Preserve a result only together
with its environment, command, date, and an explanation of the runtime being measured.

The default is 100 measured samples per scenario. A smaller run is only a smoke test. For a real
adapter, use the same sampling rules but record provider, model/deployment, region, SDK versions,
input/output sizes, rate limits, retry policy, and whether connections were already warm.

## Methodology

Each measured sample:

1. creates `--jobs` independent keys;
2. creates a fresh synthetic adapter whose invocation takes `--adapter-delay-ms`;
3. runs the selected `map`, `nested`, or complete `full` workload;
4. measures the complete batch with `time.perf_counter()`;
5. computes throughput as `jobs / batch_duration`.

Scenarios execute in the supplied concurrency order. Unmeasured warmups run before every
scenario. The output includes min, mean, p50, p95, p99, and max. Percentiles use linear
interpolation (R-7, the common NumPy default). p95/p99 describe the distribution of complete
batch durations across samples; they are not calculated from one run.

`map` uses one fragment per key. `nested` uses `--fragments` fragments and includes per-key
consolidation. `full` adds the final reduce call. Every result records the expected and observed
adapter-call count.

Resource measurement is a separate, un-timed probe so `tracemalloc` and task inspection do not
distort the latency distributions. It uses `--resource-jobs` (1000 by default), reporting peak
traced Python allocations and peak pending asyncio tasks above the pre-probe baseline.

## Interpreting the metrics

### Throughput

`throughput_jobs_s` is the primary result. Compare its p50 and tail behavior across concurrency
limits. The JSON also reports p50 throughput speedup relative to the first scenario, which should
normally be concurrency `1`.

Throughput can stop scaling because of provider rate limits, connection pools, CPU, serialization,
memory, or the adapter itself. The deterministic adapter isolates PromptNest scheduling from those
external effects.

### p95 and p99 latency

`total_latency_ms` describes whole-batch latency. p95 and p99 require enough independent samples:
20 is a quick smoke benchmark, while 50–100 or more is more credible for tail analysis. Report
the complete distribution and sample count rather than only the mean.

`adapter_call_latency_ms` measures time spent inside the synthetic adapter after admission through
PromptNest's concurrency semaphore. It intentionally excludes time waiting for that semaphore.

For independent one-fragment `map` jobs, `orchestration_overhead_ms` is:

```text
observed duration − ceil(jobs / concurrency) × configured adapter delay
```

This is not reported for nested/full workloads because their dependency graph permits overlap
between map and consolidation, so the simple expression would not be a defensible theoretical
baseline. Set `--adapter-delay-ms 0` for a second view of pure scheduler/validation cost.

### “Low latency”

The repository makes no absolute low-latency claim. Such a claim needs a comparator or SLO.
Valid comparisons include:

- concurrency `1` versus a chosen bounded concurrency;
- PromptNest versus a named orchestrator with the same adapter and workload;
- observed p95/p99 versus a declared service-level objective.

Do not call the synthetic 5 ms workload representative of a real LLM. It exists to make scheduler
behavior fast, deterministic, and reproducible.

### TTFT

TTFT is supported when an adapter implements `StreamingLLMAdapter`. Enable it with
`set_streaming(on_delta=...)`. PromptNest reports nearest-rank p50/p95/p99 for time from adapter
stream start to the first non-empty text delta, complete-stream duration, and inter-delta gaps.
Each observation retains provider, stage, key, attempt and fragment context.

The metric is precisely a **first non-empty provider text-delta latency**. Providers may group or
split tokenizer tokens, so PromptNest does not claim that every delta is exactly one model token.
The deterministic certificate validates the measurement machinery with synthetic streams. Use
`docs/benchmarks/openai_provider.py` for real OpenAI model/network distributions.

Retries are allowed before the first visible delta. Once a delta has reached the callback, an
interrupted stream is not retried automatically because replay could duplicate externally visible
text.

### Backpressure

PromptNest 0.2 uses a fixed worker pool and bounded queue. With `from_async()`, a producer awaits
admission when the queue is full. `execution_metrics` records capacity, high-watermark and
admission waits, while provider policies independently bound active external calls.

The legacy benchmark remains useful for comparisons, but the authoritative backpressure gates now
live in the [core certification](certification.md): 10,000 lazy jobs, queue capacity 128, bounded
asyncio task count, memory ceiling and observable producer blocking.

## Failures under load

`docs/benchmarks/resilience.py` deterministically verifies four 100-job scenarios:

- every job fails its first attempt and succeeds on retry;
- adapter calls exceed their timeout and are cleaned up;
- half of the jobs fail while successful partial results are retained;
- the caller cancels an active batch and adapter calls are cancelled.

The command exits non-zero if any scenario fails and records durations, retries, cancellations,
success/failure counts, and active calls remaining after cleanup.

## Automated benchmark tests

The benchmark implementation is tested independently of PromptNest's normal unit tests:

```fish
uv run pytest tests/test_performance_benchmark.py
```

These tests cover R-7 percentile interpolation, JSON schema fields, speedup calculation,
`max_active_calls`, map/nested/full call counts, overhead, resource fields, invalid CLI arguments,
output-file writing, and all failure-under-load scenarios.

## Production benchmark checklist

Keep these fixed or record them alongside every result:

- commit SHA and clean/dirty working-tree state;
- Python, OS, CPU, and memory;
- adapter, provider, model/deployment, region, and SDK versions;
- job count, fragment sizes, output schema, and expected output size;
- concurrency, connection-pool size, timeout, retry, and rate-limit settings;
- warm/cold connection state and warmup policy;
- sample count, percentile method, errors, retries, and cancellations;
- jobs/s, batch latency p50/p95/p99, and peak memory.

The JSON output records generation time, exact command, total benchmark duration, commit SHA,
dirty/clean state, Python/platform, CPU, logical CPU count, physical memory, error count, retry
count, workload shape, and percentile method.

Run concurrency scenarios separately when using a remote provider so one scenario's throttling or
connection state does not contaminate the next. Randomize or rotate scenario order when making a
comparative claim.
