"""Reproducible PromptNest orchestration benchmark.

Run from the repository root:

    uv run python docs/benchmarks/performance.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from promptnest import PromptNest

ResultModel = TypeVar("ResultModel", bound=BaseModel)
Workload = Literal["map", "nested", "full"]


class BenchmarkResult(BaseModel):
    value: str


class SyntheticAdapter:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.active = 0
        self.max_active = 0
        self.call_count = 0
        self.call_durations_ms: list[float] = []

    async def invoke(
        self,
        prompt: str,
        output_model: type[ResultModel],
        **kwargs: Any,
    ) -> ResultModel:
        del kwargs
        started = time.perf_counter()
        self.call_count += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            return output_model.model_validate({"value": prompt})
        finally:
            self.active -= 1
            self.call_durations_ms.append((time.perf_counter() - started) * 1_000)


@dataclass(frozen=True)
class ResourceProbe:
    jobs: int
    peak_traced_memory_bytes: int
    peak_pending_tasks_above_baseline: int


@dataclass(frozen=True)
class Scenario:
    workload: Workload
    concurrency: int
    samples: int
    jobs_per_sample: int
    fragments_per_job: int
    adapter_calls_per_sample: int
    total_latency_ms: dict[str, float]
    throughput_jobs_s: dict[str, float]
    adapter_call_latency_ms: dict[str, float]
    orchestration_overhead_ms: dict[str, float] | None
    max_active_calls: int
    resources: ResourceProbe


def percentile(values: list[float], probability: float) -> float:
    """Return a linearly interpolated percentile (R-7 / NumPy default)."""
    if not values or not 0 <= probability <= 1:
        raise ValueError("values must be non-empty and probability must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 3),
        "p50": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
        "p99": round(percentile(values, 0.99), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.fmean(values), 3),
    }


def expected_calls(jobs: int, fragments: int, workload: Workload) -> int:
    calls = jobs * fragments
    if fragments > 1:
        calls += jobs
    if workload == "full":
        calls += 1
    return calls


def build_runner(
    adapter: SyntheticAdapter,
    *,
    jobs: int,
    fragments: int,
    concurrency: int,
    workload: Workload,
) -> PromptNest[int, BenchmarkResult, BenchmarkResult]:
    chunks = {
        key: [f"job-{key}-fragment-{fragment}" for fragment in range(fragments)]
        for key in range(jobs)
    }
    runner = (
        PromptNest.have(adapter, chunks)
        .set_concurrency(concurrency)
        .set_retry_config(max_attempts=1, delay_s=0, timeout_s=30)
        .set_pre_prompt("{chunk_text}", BenchmarkResult)
    )
    if workload == "full":
        runner = runner.set_pos_prompt("{partial_answers}", BenchmarkResult)
    return runner


async def execute_runner(
    runner: PromptNest[int, BenchmarkResult, BenchmarkResult],
    workload: Workload,
) -> None:
    await runner.get_chunks_result()
    if workload == "full":
        await runner.run_pos_prompt()


async def one_sample(
    *,
    jobs: int,
    fragments: int,
    concurrency: int,
    delay_s: float,
    workload: Workload,
) -> tuple[float, SyntheticAdapter]:
    adapter = SyntheticAdapter(delay_s)
    runner = build_runner(
        adapter,
        jobs=jobs,
        fragments=fragments,
        concurrency=concurrency,
        workload=workload,
    )
    started = time.perf_counter()
    await execute_runner(runner, workload)
    return time.perf_counter() - started, adapter


async def resource_probe(
    *,
    jobs: int,
    fragments: int,
    concurrency: int,
    delay_s: float,
    workload: Workload,
) -> ResourceProbe:
    adapter = SyntheticAdapter(delay_s)
    runner = build_runner(
        adapter,
        jobs=jobs,
        fragments=fragments,
        concurrency=concurrency,
        workload=workload,
    )
    baseline = len([task for task in asyncio.all_tasks() if not task.done()])
    peak_pending = 0
    tracemalloc.start()
    task = asyncio.create_task(execute_runner(runner, workload))
    while not task.done():
        pending = len([item for item in asyncio.all_tasks() if not item.done()])
        peak_pending = max(peak_pending, pending - baseline)
        await asyncio.sleep(0.001)
    await task
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return ResourceProbe(
        jobs=jobs,
        peak_traced_memory_bytes=peak_memory,
        peak_pending_tasks_above_baseline=peak_pending,
    )


async def run_scenario(
    *,
    jobs: int,
    fragments: int,
    concurrency: int,
    samples: int,
    warmups: int,
    delay_s: float,
    workload: Workload,
    resource_jobs: int,
) -> Scenario:
    sample_options = {
        "jobs": jobs,
        "fragments": fragments,
        "concurrency": concurrency,
        "delay_s": delay_s,
        "workload": workload,
    }
    for _ in range(warmups):
        await one_sample(**sample_options)

    durations_s: list[float] = []
    call_durations_ms: list[float] = []
    max_active = 0
    calls = expected_calls(jobs, fragments, workload)
    for _ in range(samples):
        duration_s, adapter = await one_sample(**sample_options)
        if adapter.call_count != calls:
            raise RuntimeError(f"expected {calls} adapter calls, observed {adapter.call_count}")
        durations_s.append(duration_s)
        call_durations_ms.extend(adapter.call_durations_ms)
        max_active = max(max_active, adapter.max_active)

    overhead: dict[str, float] | None = None
    if workload == "map" and fragments == 1:
        theoretical_s = math.ceil(jobs / concurrency) * delay_s
        overhead = distribution(
            [max(0.0, duration - theoretical_s) * 1_000 for duration in durations_s]
        )

    return Scenario(
        workload=workload,
        concurrency=concurrency,
        samples=samples,
        jobs_per_sample=jobs,
        fragments_per_job=fragments,
        adapter_calls_per_sample=calls,
        total_latency_ms=distribution([value * 1_000 for value in durations_s]),
        throughput_jobs_s=distribution([jobs / value for value in durations_s]),
        adapter_call_latency_ms=distribution(call_durations_ms),
        orchestration_overhead_ms=overhead,
        max_active_calls=max_active,
        resources=await resource_probe(
            **{
                **sample_options,
                "jobs": resource_jobs,
            }
        ),
    )


def parse_csv_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",")]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("value must contain positive integers")
    return values


def parse_workloads(value: str) -> list[Workload]:
    workloads = value.split(",")
    allowed = {"map", "nested", "full"}
    if not workloads or any(item not in allowed for item in workloads):
        raise argparse.ArgumentTypeError("workloads must contain map, nested, or full")
    return [item for item in workloads if item in allowed]  # type: ignore[misc]


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=100)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--fragments", type=int, default=3)
    parser.add_argument("--resource-jobs", type=int, default=1000)
    parser.add_argument("--concurrency", type=parse_csv_ints, default=parse_csv_ints("1,4,16"))
    parser.add_argument(
        "--workloads",
        type=parse_workloads,
        default=parse_workloads("map,nested,full"),
    )
    parser.add_argument("--adapter-delay-ms", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (
        args.jobs < 1
        or args.samples < 1
        or args.warmups < 0
        or args.fragments < 2
        or args.resource_jobs < 1
        or args.adapter_delay_ms < 0
    ):
        parser.error(
            "jobs/samples/resource-jobs must be positive, fragments >= 2, "
            "and warmups/delay non-negative"
        )
    return args


def git_metadata() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            check=False,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "commit_sha": git("rev-parse", "HEAD"),
        "working_tree": None if status is None else ("clean" if not status else "dirty"),
    }


def host_metadata() -> dict[str, Any]:
    cpu = platform.processor() or None
    try:
        cpu_lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        cpu = next(
            line.split(":", 1)[1].strip()
            for line in cpu_lines
            if line.lower().startswith("model name")
        )
    except (OSError, StopIteration):
        pass
    page_size = os.sysconf("SC_PAGE_SIZE")
    physical_pages = os.sysconf("SC_PHYS_PAGES")
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_bytes": page_size * physical_pages,
    }


def serialize_scenarios(scenarios: list[Scenario]) -> list[dict[str, Any]]:
    baselines: dict[Workload, float] = {}
    serialized: list[dict[str, Any]] = []
    for scenario in scenarios:
        throughput = scenario.throughput_jobs_s["p50"]
        baseline = baselines.setdefault(scenario.workload, throughput)
        serialized.append(
            {
                **asdict(scenario),
                "p50_throughput_speedup_vs_first_concurrency": round(
                    throughput / baseline,
                    3,
                ),
            }
        )
    return serialized


async def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    started_wall = datetime.now(UTC)
    started = time.perf_counter()
    scenarios = []
    for workload in args.workloads:
        fragments = 1 if workload == "map" else args.fragments
        for concurrency in args.concurrency:
            scenarios.append(
                await run_scenario(
                    jobs=args.jobs,
                    fragments=fragments,
                    concurrency=concurrency,
                    samples=args.samples,
                    warmups=args.warmups,
                    delay_s=args.adapter_delay_ms / 1_000,
                    workload=workload,
                    resource_jobs=args.resource_jobs,
                )
            )
    duration_s = time.perf_counter() - started
    return {
        "schema_version": 2,
        "generated_at": started_wall.isoformat(),
        "command": [sys.executable, *sys.argv],
        "duration_s": round(duration_s, 3),
        "environment": {**host_metadata(), **git_metadata()},
        "methodology": {
            "clock": "time.perf_counter",
            "percentile_method": "linear interpolation (R-7)",
            "adapter_delay_ms": args.adapter_delay_ms,
            "warmups_per_scenario": args.warmups,
            "errors": 0,
            "retries": 0,
            "resource_probe": (
                "one separate run per scenario; excluded from latency distributions; "
                "memory measured with tracemalloc"
            ),
            "overhead": (
                "observed batch duration - ceil(jobs/concurrency) * adapter delay; "
                "reported only for independent one-fragment map jobs"
            ),
        },
        "scope": {
            "ttft": {
                "supported": False,
                "reason": "LLMAdapter.invoke exposes a complete structured result, not a stream",
            },
            "backpressure": {
                "supported": False,
                "bounded_concurrency": True,
                "reason": (
                    "active calls are bounded, but pending work grows with the complete input batch"
                ),
            },
            "low_latency_claim": "requires an explicit SLO or named comparator",
        },
        "scenarios": serialize_scenarios(scenarios),
    }


def write_result(payload: dict[str, Any], output: Path | None) -> str:
    rendered = json.dumps(payload, indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{rendered}\n", encoding="utf-8")
    return rendered


async def main() -> None:
    args = arguments()
    payload = await benchmark(args)
    print(write_result(payload, args.output))


if __name__ == "__main__":
    asyncio.run(main())
